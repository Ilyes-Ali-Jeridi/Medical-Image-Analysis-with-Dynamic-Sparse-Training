import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModel, AutoTokenizer
import logging
import faiss # For similarity search in RAG
import numpy as np

logger = logging.getLogger(__name__)

class DynamicSparseLayer(nn.Module):
    """
    A wrapper layer that introduces dynamic sparsity to a given base layer.

    This layer maintains a binary mask that is applied to the base layer's weights
    during the forward pass. The mask is periodically updated based on the
    magnitude of weight gradients, effectively pruning and regrowing connections.

    Attributes:
        base_layer (nn.Module): The PyTorch layer (e.g., nn.Linear, nn.Conv2d) to which sparsity is applied.
        sparsity (float): The target fraction of weights to be zeroed out (pruned). Must be between 0 and 1.
        update_freq (int): How often (in training steps) the sparsity mask should be updated.
        mask (torch.Tensor): The binary mask applied to the weights.
        step_counter (int): Counter for tracking training steps for mask updates.
    """
    def __init__(self, base_layer: nn.Module, sparsity: float = 0.3, update_freq: int = 100):
        """
        Initializes the DynamicSparseLayer.

        Args:
            base_layer (nn.Module): The layer to wrap (must have a 'weight' attribute).
            sparsity (float, optional): Target sparsity level. Defaults to 0.3 (30% sparsity).
            update_freq (int, optional): Frequency of mask updates. Defaults to 100 steps.

        Raises:
            ValueError: If sparsity is not between 0 and 1.
            AttributeError: If `base_layer` does not have a 'weight' attribute.
        """
        super().__init__()
        if not 0.0 <= sparsity < 1.0: # Sparsity must be less than 1
            raise ValueError(f"Sparsity must be between 0 (inclusive) and 1 (exclusive), got {sparsity}")
            
        self.base_layer = base_layer
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.step_counter = 0 # Counter for training steps to trigger mask updates
        
        # Initialize the mask based on initial weight magnitudes
        try:
            with torch.no_grad(): # Ensure no gradients are computed during mask initialization
                if not hasattr(self.base_layer, 'weight') or self.base_layer.weight is None:
                     raise AttributeError(f"Base layer {type(self.base_layer)} must have a 'weight' attribute.")
                weight = self.base_layer.weight.data # Get the actual weight tensor
                # Calculate the threshold for pruning based on weight magnitudes
                threshold = torch.quantile(weight.abs().view(-1), self.sparsity)
                # Create and register the binary mask buffer
                self.register_buffer('mask', (weight.abs() > threshold).float())
                logger.info(f"Initialized DynamicSparseLayer for {type(base_layer)} with sparsity {sparsity:.2f}. Initial mask density: {self.mask.mean():.2f}")
        except AttributeError as ae: # If base_layer doesn't have 'weight'
            logger.error(f"Failed to initialize sparse layer: base_layer {type(self.base_layer)} missing 'weight' attribute. Error: {ae}", exc_info=True)
            raise
        except Exception as e: # General catch-all
            logger.error(f"Failed to initialize sparse layer for {type(self.base_layer)}: {e}", exc_info=True)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the sparsified base layer.

        The base layer's weights are masked before the operation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the sparsified base layer.
        
        Raises:
            RuntimeError: If tensor operations fail (e.g., shape mismatch).
            Exception: For other unexpected errors.
        """
        try:
            # Apply the mask to the base layer's weights to get effective weights
            effective_weight = self.base_layer.weight * self.mask
            
            # Perform the appropriate operation based on the type of the base layer
            if isinstance(self.base_layer, nn.Conv2d):
                return F.conv2d(x, effective_weight, self.base_layer.bias,
                                self.base_layer.stride, self.base_layer.padding,
                                self.base_layer.dilation, self.base_layer.groups)
            elif isinstance(self.base_layer, nn.Linear):
                return F.linear(x, effective_weight, self.base_layer.bias)
            else:
                # If the layer type is not explicitly handled, log a warning.
                # Sparsity might not be correctly applied if the layer doesn't use 'weight' in a standard way.
                logger.warning(
                    f"DynamicSparseLayer applying to an unsupported layer type {type(self.base_layer)}. "
                    "Sparsity relies on direct weight multiplication. Ensure this is appropriate."
                )
                # Fallback: attempt to call the base layer directly.
                # This might not use the effective_weight if the base_layer's forward doesn't use self.weight.
                return self.base_layer(x) 
        except RuntimeError as rt_e:
            logger.error(f"Runtime error in DynamicSparseLayer forward pass for {type(self.base_layer)}: {rt_e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DynamicSparseLayer forward pass for {type(self.base_layer)}: {e}", exc_info=True)
            raise

    def update_mask(self):
        """
        Updates the sparsity mask based on gradient importance (magnitude of weight * gradient).

        This method is typically called during the training loop. It identifies weights
        with the largest combined magnitude of value and gradient and keeps them,
        pruning the rest to maintain the target sparsity.
        
        Raises:
            RuntimeError: If tensor operations during mask update fail.
            Exception: For other unexpected errors.
        """
        try:
            self.step_counter += 1
            # Check if it's time to update and if gradients are available
            if self.step_counter % self.update_freq == 0 and \
               hasattr(self.base_layer.weight, 'grad') and \
               self.base_layer.weight.grad is not None:
                
                with torch.no_grad(): # Mask update should not be part of gradient computation
                    weight = self.base_layer.weight
                    grad = self.base_layer.weight.grad
                    
                    # Calculate score for keeping weights: absolute value of (weight * gradient)
                    # This score prioritizes weights that are both large and have large gradients.
                    score = (weight * grad).abs()
                    flat_score = score.view(-1) # Flatten scores to rank them
                    
                    num_params = flat_score.numel() # Total number of parameters
                    num_keep = int(num_params * (1.0 - self.sparsity)) # Number of parameters to keep
                    
                    if num_keep < 1: # Ensure at least one weight is kept if sparsity is very high
                        logger.warning(f"Sparsity level {self.sparsity} is too high for layer {type(self.base_layer)}, resulting in num_keep < 1. Keeping minimal weights or consider adjusting sparsity.")
                        # Option: keep top-1 if num_keep is 0, or create an all-zero mask if that's intended.
                        # For now, creates an all-zero mask if num_keep is < 1.
                        new_mask = torch.zeros_like(self.mask)
                    else:
                        # Find the top-k scores (indices of weights to keep)
                        _, topk_indices = torch.topk(flat_score, num_keep, sorted=False)
                        
                        # Create a new mask and set the kept weights to 1.0
                        new_mask = torch.zeros_like(flat_score)
                        new_mask[topk_indices] = 1.0
                        new_mask = new_mask.view_as(self.mask) # Reshape to original weight dimensions
                    
                    self.mask.copy_(new_mask) # Update the layer's mask
                    logger.debug(f"Updated sparsity mask for {type(self.base_layer)}. New mask density: {self.mask.mean():.2f}")
        except RuntimeError as rt_e:
            logger.error(f"Runtime error during mask update for {type(self.base_layer)}: {rt_e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during mask update for {type(self.base_layer)}: {e}", exc_info=True)
            raise

class MedicalViT(nn.Module):
    """
    Vision Transformer (ViT) specifically adapted for medical imaging tasks.

    This model processes medical images (e.g., X-rays) using a transformer
    architecture. It includes patch embedding (optionally sparse), positional
    embeddings, a series of transformer encoder blocks, and a classification head.

    Attributes:
        patch_embed (nn.Module): Layer to convert image patches into embeddings. Can be a DynamicSparseLayer.
        pos_embed (nn.Parameter): Learnable positional embeddings for patch sequences.
        blocks (nn.ModuleList): List of transformer encoder layers.
        norm (nn.LayerNorm): Layer normalization applied after transformer blocks.
        classifier_head (nn.Sequential): Final classification head.
    """
    def __init__(self, sparse_rate: float = 0.4, img_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 8):
        """
        Initializes the MedicalViT model.

        Args:
            sparse_rate (float, optional): Sparsity rate for patch embedding and classifier head's linear layer if DynamicSparseLayer is used. Defaults to 0.4.
            img_size (int, optional): Size of the input image (assumed square). Defaults to 256.
            patch_size (int, optional): Size of patches the image is divided into. Defaults to 16.
            embed_dim (int, optional): Dimensionality of embeddings. Defaults to 768.
            num_layers (int, optional): Number of transformer encoder layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads in transformer layers. Defaults to 8.
        """
        super().__init__()
        logger.info(f"Initializing MedicalViT: img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}, layers={num_layers}, heads={num_heads}, sparse_rate={sparse_rate}")

        # Patch embedding layer: converts image patches (e.g., 16x16) into flat embeddings.
        # Uses DynamicSparseLayer for potential efficiency and regularization.
        self.patch_embed = DynamicSparseLayer(
            nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size), # Assuming 1 input channel (grayscale)
            sparsity=sparse_rate
        )
        
        # Learnable positional embeddings for each patch.
        # Shape: (1, num_patches, embed_dim) where num_patches = (img_size // patch_size) ** 2
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # Initialize positional embeddings with a normal distribution.
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,          # Input feature dimension
                nhead=num_heads,            # Number of attention heads
                dim_feedforward=embed_dim * 4, # Dimension of the feed-forward network
                dropout=0.1,                # Dropout rate
                activation=F.gelu,          # Activation function
                batch_first=True,           # Input/output tensors are (batch, seq, feature)
                norm_first=True             # Apply layer norm before self-attention and feed-forward (Pre-LN)
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization after transformer blocks
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head: processes the aggregated features from the transformer
        # to produce logits for classification (e.g., 5 classes for certain findings).
        # Includes a sparse linear layer.
        self.classifier_head = nn.Sequential(
            DynamicSparseLayer(nn.Linear(embed_dim, embed_dim // 3), sparsity=sparse_rate),
            nn.GELU(),      # GELU activation function
            nn.Dropout(0.1), # Dropout for regularization
            nn.Linear(embed_dim // 3, 5) # Output layer for 5 classes (example)
        )
        
    def forward(self, x: torch.Tensor, return_both: bool = False, return_features_only: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MedicalViT.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W), where C is typically 1 for grayscale.
            return_both (bool, optional): If True, returns both features (after ViT mean pooling) and classification logits. Defaults to False.
            return_features_only (bool, optional): If True, returns only the features (after ViT mean pooling). Defaults to False.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: 
                - If return_features_only is True: image features tensor.
                - If return_both is True: a tuple (features, logits).
                - Otherwise: classification logits tensor.
        """
        B = x.shape[0] # Batch size
        
        # Patch embedding and positional encoding
        x = self.patch_embed(x)           # (B, embed_dim, H_patch, W_patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed            # Add positional embeddings
        
        # Pass through transformer encoder blocks
        for block in self.blocks:
            x = block(x)
            
        # Global average pooling (mean of patch embeddings) and layer normalization
        # This aggregates information across all patches.
        features = self.norm(x.mean(dim=1)) # (B, embed_dim)
        
        # Return based on flags
        if return_features_only:
            return features
        
        logits = self.classifier_head(features) # (B, num_classes)
        if return_both:
            return features, logits
        else:
            return logits

class MedicalRAG(nn.Module):
    """
    Retrieval-Augmented Generation (RAG) model for medical report generation.

    This model combines a vision encoder (MedicalViT) with a language model (BART)
    and a text encoder (RadBERT) for retrieval. It generates medical reports
    conditioned on input images and, optionally, retrieved relevant reports.

    Key Components:
        vision_encoder (MedicalViT): Encodes input images into feature representations.
        language_model (BartForConditionalGeneration): Generates report text.
        tokenizer (BartTokenizer): Tokenizer for the language model.
        text_encoder (AutoModel): Encodes text reports for building a retrieval index (e.g., RadBERT).
        text_tokenizer (AutoTokenizer): Tokenizer for the text encoder.
        retrieval_index (faiss.Index, optional): FAISS index for efficient similarity search of report embeddings.
        report_db (List[str], optional): List of report texts corresponding to embeddings in the FAISS index.
        context_proj (nn.Sequential): Projection layer to fuse image and retrieved context features.
    """
    def __init__(self, vision_encoder: MedicalViT = None, sparse_rate: float = 0.4):
        """
        Initializes the MedicalRAG model.

        Args:
            vision_encoder (MedicalViT, optional): An instance of MedicalViT. If None, a new MedicalViT
                                                   is initialized with the given sparse_rate. Defaults to None.
            sparse_rate (float, optional): Sparsity rate used if a new MedicalViT is initialized. Defaults to 0.4.
        
        Raises:
            OSError: If loading pretrained models (BART, RadBERT) from HuggingFace Hub fails.
            Exception: For other unexpected errors during initialization.
        """
        super().__init__()
        logger.info(f"Initializing MedicalRAG model. Vision encoder provided: {vision_encoder is not None}, Sparse rate for default ViT: {sparse_rate}")
        try:
            # Initialize the vision encoder (e.g., MedicalViT)
            self.vision_encoder = vision_encoder if vision_encoder is not None else MedicalViT(sparse_rate=sparse_rate)
            
            # Load pretrained language model (BART) and its tokenizer
            logger.info("Loading BART language model (facebook/bart-base) and tokenizer...")
            self.language_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            logger.info("BART model and tokenizer loaded successfully.")

            # Load pretrained text encoder (RadBERT) and its tokenizer for retrieval purposes
            logger.info("Loading RadBERT text encoder (allenai/radbert) and tokenizer for retrieval...")
            self.text_encoder = AutoModel.from_pretrained("allenai/radbert")
            self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/radbert")
            logger.info("RadBERT model and tokenizer loaded successfully.")

        except OSError as ose: # Handles issues like model not found, network problems
            logger.error(f"Failed to load a pretrained HuggingFace model/tokenizer for MedicalRAG: {ose}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other initialization errors
            logger.error(f"Unexpected error during MedicalRAG initialization: {e}", exc_info=True)
            raise
        
        # Projection layer to combine image features and retrieved context features
        # Input dimension is embed_dim (image) + embed_dim (context) = 768*2 if embed_dim is 768.
        # Output dimension is embed_dim, to match BART's expected input.
        self.context_proj = nn.Sequential(
            nn.Linear(self.vision_encoder.classifier_head[0].base_layer.in_features * 2, self.vision_encoder.classifier_head[0].base_layer.in_features), # Assuming embed_dim is accessible this way
            nn.LayerNorm(self.vision_encoder.classifier_head[0].base_layer.in_features),
            nn.GELU(),
            nn.Dropout(0.1) # Regularization
        )
        
        # FAISS index for report retrieval (initialized as None, built later)
        self.retrieval_index = None
        # Database of reports corresponding to the FAISS index (populated during build_index)
        self.report_db = [] # This needs to be populated for retrieve_context to work with text
        
    def build_index(self, dataset_loader: DataLoader):
        """
        Builds a FAISS index from report embeddings in the provided dataset_loader.

        This method iterates through the dataset, encodes reports using `self.text_encoder`,
        and adds the resulting embeddings to a FAISS index for efficient similarity search.
        The `self.report_db` should also be populated here if it's used by `retrieve_context`.

        Args:
            dataset_loader (DataLoader): DataLoader providing batches of data,
                                         where each batch should contain a 'report' key with report texts.
        
        Raises:
            AttributeError: If `self.text_encoder` is not properly initialized.
            RuntimeError: For issues during FAISS index building or tensor operations (e.g., OOM).
        """
        if not hasattr(dataset_loader, 'dataset') or not dataset_loader.dataset:
            logger.warning("FAISS index build: dataset_loader is empty or has no dataset. Index will not be built.")
            self.retrieval_index = None
            return

        logger.info(f"Building FAISS index from {len(dataset_loader.dataset)} reports...")
        all_embeds = [] # List to store all report embeddings
        self.report_db = [] # Clear or initialize report_db for the current index build
        
        try:
            device = next(self.text_encoder.parameters()).device # Get device of text_encoder
            
            for i, batch in enumerate(dataset_loader): # Iterate through dataset
                reports = batch.get('report', []) # Get reports from batch
                if not reports:
                    logger.debug(f"Batch {i} in dataset_loader has no 'report' data. Skipping for FAISS index.")
                    continue
                
                # Store reports in self.report_db if they are to be retrieved by index later
                self.report_db.extend(reports) 

                # Tokenize and encode reports
                inputs = self.text_tokenizer(
                    reports,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512 # Max length for RadBERT, consider making configurable
                ).to(device)
                
                with torch.no_grad(): # No gradients needed for encoding
                    outputs = self.text_encoder(**inputs)
                    # Use mean of last hidden states as report embedding
                    embeds = outputs.last_hidden_state.mean(dim=1) # (batch_size, embed_dim)
                    all_embeds.append(embeds.cpu().numpy()) # Store embeddings as NumPy arrays on CPU
            
            if not all_embeds:
                logger.warning("No embeddings were generated during FAISS index build. Index will be empty.")
                self.retrieval_index = None
                return

            all_embeds_np = np.concatenate(all_embeds, axis=0) # Combine all batch embeddings
            d = all_embeds_np.shape[1] # Dimension of embeddings
            
            logger.info(f"Creating FAISS index with {all_embeds_np.shape[0]} embeddings of dimension {d}.")
            # Using IndexFlatIP for Inner Product similarity (cosine similarity after normalization)
            self.retrieval_index = faiss.IndexFlatIP(d) 
            faiss.normalize_L2(all_embeds_np) # Normalize embeddings for cosine similarity
            self.retrieval_index.add(all_embeds_np) # Add embeddings to the index
            logger.info(f"FAISS index built successfully with {self.retrieval_index.ntotal} entries.")

        except StopIteration:
            logger.warning("Dataset loader for FAISS index building was empty or exhausted prematurely. Index may be incomplete or not built.")
            self.retrieval_index = None
        except AttributeError as ae:
            logger.error(f"Attribute error during FAISS index build (e.g., text_encoder not initialized?): {ae}", exc_info=True)
            self.retrieval_index = None
            raise 
        except RuntimeError as rt_e:
            logger.error(f"Runtime error during FAISS index build (e.g., OOM, device mismatch): {rt_e}", exc_info=True)
            self.retrieval_index = None
        except Exception as e:
            logger.error(f"Unexpected error during FAISS index build: {e}", exc_info=True)
            self.retrieval_index = None
        
    def retrieve_context(self, img_embed: torch.Tensor, k: int = 3) -> List[str]:
        """
        Retrieves k-most relevant reports from the FAISS index based on image embeddings.

        Args:
            img_embed (torch.Tensor): Batch of image embeddings (B, embed_dim).
            k (int, optional): Number of relevant reports to retrieve for each image. Defaults to 3.

        Returns:
            List[str]: A list of concatenated context strings (k reports joined by space),
                       one for each image embedding in the batch. Returns empty strings if retrieval fails.
        """
        if self.retrieval_index is None:
            logger.warning("Retrieval index is not built. Cannot retrieve context. Returning empty context strings.")
            return [""] * img_embed.size(0)
        if not self.report_db: # self.report_db should be populated by build_index
            logger.warning("Report database (self.report_db) is empty. Cannot retrieve context strings. Returning empty context strings.")
            return [""] * img_embed.size(0)
            
        try:
            # Normalize image embeddings and convert to NumPy for FAISS
            img_embed_np = img_embed.detach().cpu().numpy() # Ensure on CPU and NumPy
            faiss.normalize_L2(img_embed_np) # Normalize for cosine similarity search with IndexFlatIP
            
            logger.debug(f"Searching FAISS index for {k} nearest neighbors for {img_embed_np.shape[0]} queries.")
            # Search the FAISS index for k nearest neighbors
            scores, indices = self.retrieval_index.search(img_embed_np, k) # scores are L2 distances or IP
            
            context_reports = []
            for i, sample_indices in enumerate(indices): # For each query image
                # Filter out invalid indices (e.g., -1 if k > ntotal or search issues)
                # and ensure indices are within the bounds of report_db.
                valid_indices = [idx for idx in sample_indices if 0 <= idx < len(self.report_db)]
                
                if len(valid_indices) < k and len(valid_indices) > 0 : # Log if fewer than k reports found but some were found
                     logger.debug(f"Retrieved {len(valid_indices)} reports (less than k={k}) for query {i}.")
                elif len(valid_indices) == 0:
                     logger.debug(f"No valid reports retrieved for query {i}.")


                # Join the retrieved reports to form a single context string
                context = " ".join([self.report_db[idx] for idx in valid_indices])
                context_reports.append(context)
            
            return context_reports
        except RuntimeError as faiss_rt_e:
            logger.error(f"FAISS search failed during context retrieval: {faiss_rt_e}", exc_info=True)
            return [""] * img_embed.size(0) # Fallback to empty contexts
        except Exception as e:
            logger.error(f"Unexpected error during context retrieval: {e}", exc_info=True)
            return [""] * img_embed.size(0) # Fallback
    
    def forward(self, images: torch.Tensor, target_reports: Optional[List[str]] = None, 
                precomputed_features: Optional[torch.Tensor] = None) -> Dict[str, Any] | List[str]:
        """
        Forward pass of the MedicalRAG model.

        Args:
            images (torch.Tensor): Batch of input images (B, C, H, W).
            target_reports (Optional[List[str]], optional): List of target report strings for training (teacher forcing).
                                                            If None, operates in inference/generation mode. Defaults to None.
            precomputed_features (Optional[torch.Tensor], optional): Optional precomputed image features.
                                                                     If provided, `images` might be ignored for feature extraction.
                                                                     Defaults to None.

        Returns:
            Dict[str, Any] | List[str]:
                - If `target_reports` is provided (training/validation mode):
                  A dictionary with "loss" (torch.Tensor) and "generated_reports" (List[str]).
                - If `target_reports` is None (inference mode):
                  A list of generated report strings.
                Returns error-specific outputs if exceptions occur.
        """
        try:
            # Step 1: Obtain image features (either precomputed or via vision_encoder)
            if precomputed_features is None:
                if self.vision_encoder is None:
                    logger.error("Vision encoder is not initialized in MedicalRAG. Cannot perform forward pass.")
                    raise ValueError("Vision encoder is not initialized for MedicalRAG.")
                img_features = self.vision_encoder(images, return_features_only=True)
            else:
                img_features = precomputed_features
            logger.debug(f"Image features obtained. Shape: {img_features.shape}")

            # Step 2: Retrieve context reports based on image features
            # This step is skipped if retrieval_index is not built.
            context_reports = self.retrieve_context(img_features)
            logger.debug(f"Retrieved {len(context_reports)} context reports. Example: '{context_reports[0][:100]}...'")
            
            # Step 3: Tokenize context reports and get their embeddings via language_model's encoder
            # This is done without gradients as context is fixed during this pass.
            with torch.no_grad():
                context_tokens = self.tokenizer(
                    context_reports,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128, # Max length for context tokens, consider making configurable
                    padding=True    # Pad to longest in batch
                ).to(images.device) # Ensure tokens are on the same device as images/model
                
                # Get embeddings for the context using the encoder part of the language model
                context_outputs = self.language_model.model.encoder(
                    input_ids=context_tokens.input_ids,
                    attention_mask=context_tokens.attention_mask
                )
                context_embeds = context_outputs.last_hidden_state.mean(dim=1) # Mean pool context token embeddings
            logger.debug(f"Context embeddings obtained. Shape: {context_embeds.shape}")
                
            # Step 4: Fuse image features and context embeddings
            fused_features = self.context_proj(torch.cat([img_features, context_embeds], dim=-1))
            # Unsqueeze to add a sequence length dimension of 1, as BART expects [batch, seq_len, hidden_dim] for inputs_embeds
            input_embeds_for_bart = fused_features.unsqueeze(1)
            logger.debug(f"Fused features prepared for BART. Shape: {input_embeds_for_bart.shape}")

            # Step 5: Generate reports using the language model
            if target_reports is not None: # Training or validation with teacher forcing
                # Tokenize target reports for calculating loss
                target_tokens = self.tokenizer(
                    target_reports,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length", # Pad to max_length for labels
                    max_length=128        # Should align with generation parameters
                ).to(images.device)
                
                # Forward pass through the full language model (encoder-decoder)
                outputs = self.language_model(
                    inputs_embeds=input_embeds_for_bart, # Use fused features as input to decoder
                    labels=target_tokens.input_ids       # Target token IDs for loss calculation
                )
                
                loss = outputs.loss
                logger.debug(f"Training/Validation mode: Loss calculated: {loss.item()}")
                
                # Optionally generate reports even during training for monitoring (can be slow)
                # Parameters for generation can be part of self.config
                gen_ids = self.language_model.generate(
                    inputs_embeds=input_embeds_for_bart,
                    max_length=self.config.get('gen_max_length', 128),
                    num_beams=self.config.get('gen_num_beams', 4),
                    temperature=self.config.get('gen_temperature', 0.9),
                    no_repeat_ngram_size=self.config.get('gen_no_repeat_ngram', 3)
                )
                generated_reports = [
                    self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in gen_ids
                ]
                logger.debug(f"Generated {len(generated_reports)} reports in training/validation mode.")
                return {"loss": loss, "generated_reports": generated_reports}
            
            else: # Inference mode (no target reports provided)
                logger.debug("Inference mode: Generating reports.")
                gen_ids = self.language_model.generate(
                    inputs_embeds=input_embeds_for_bart,
                    max_length=self.config.get('gen_max_length', 128),
                    num_beams=self.config.get('gen_num_beams', 4),
                    temperature=self.config.get('gen_temperature', 0.9),
                    no_repeat_ngram_size=self.config.get('gen_no_repeat_ngram', 3)
                )
                generated_reports = [
                    self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in gen_ids
                ]
                logger.debug(f"Generated {len(generated_reports)} reports in inference mode.")
                return generated_reports
                
        except RuntimeError as rt_e:
            logger.error(f"Runtime error in MedicalRAG forward pass: {rt_e}", exc_info=True)
            # Provide a consistent error output format based on expected return type
            if target_reports is not None:
                return {"loss": torch.tensor(float('nan'), device=images.device), 
                        "generated_reports": ["Error: Runtime error during generation."] * images.size(0)}
            else:
                return ["Error: Runtime error during generation."] * images.size(0)
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error in MedicalRAG forward pass: {e}", exc_info=True)
            if target_reports is not None:
                return {"loss": torch.tensor(float('nan'), device=images.device), 
                        "generated_reports": ["Error: Unexpected error during generation."] * images.size(0)}
            else:
                return ["Error: Unexpected error during generation."] * images.size(0)