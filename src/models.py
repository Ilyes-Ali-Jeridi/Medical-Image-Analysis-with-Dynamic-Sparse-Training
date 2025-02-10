import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModel, AutoTokenizer
import logging
import faiss
import numpy as np

logger = logging.getLogger(__name__)

class DynamicSparseLayer(nn.Module):
    """Implements dynamic sparse layer with error logging and proper initialization."""
    def __init__(self, base_layer: nn.Module, sparsity: float = 0.3, update_freq: int = 100):
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")
            
        self.base_layer = base_layer
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.step_counter = 0
        
        try:
            with torch.no_grad():
                weight = self.base_layer.weight.data
                threshold = torch.quantile(weight.abs().view(-1), self.sparsity)
                self.register_buffer('mask', (weight.abs() > threshold).float())
        except Exception as e:
            logger.error(f"Failed to initialize sparse layer: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            effective_weight = self.base_layer.weight * self.mask
            if isinstance(self.base_layer, nn.Conv2d):
                return F.conv2d(x, effective_weight, self.base_layer.bias,
                              self.base_layer.stride, self.base_layer.padding)
            elif isinstance(self.base_layer, nn.Linear):
                return F.linear(x, effective_weight, self.base_layer.bias)
            else:
                return self.base_layer(x)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def update_mask(self):
        """Updates the sparsity mask based on gradient importance."""
        try:
            self.step_counter += 1
            if self.step_counter % self.update_freq == 0 and self.base_layer.weight.grad is not None:
                with torch.no_grad():
                    weight = self.base_layer.weight
                    grad = self.base_layer.weight.grad
                    score = (weight * grad).abs()
                    flat_score = score.view(-1)
                    num_params = flat_score.numel()
                    num_keep = int(num_params * (1 - self.sparsity))
                    
                    if num_keep < 1:
                        logger.warning("Sparsity too high, all weights would be pruned")
                        new_mask = torch.zeros_like(self.mask)
                    else:
                        _, topk_indices = torch.topk(flat_score, num_keep, sorted=False)
                        new_mask = torch.zeros_like(flat_score)
                        new_mask[topk_indices] = 1.0
                        new_mask = new_mask.view_as(self.mask)
                    self.mask.copy_(new_mask)
        except Exception as e:
            logger.error(f"Mask update failed: {str(e)}")
            raise

class MedicalViT(nn.Module):
    """Vision Transformer optimized for medical imaging."""
    def __init__(self, sparse_rate=0.4, img_size=256, patch_size=16,
                 embed_dim=768, num_layers=12, num_heads=8):
        super().__init__()
        self.patch_embed = DynamicSparseLayer(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
            sparsity=sparse_rate
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier_head = nn.Sequential(
            DynamicSparseLayer(nn.Linear(embed_dim, embed_dim // 3), sparsity=sparse_rate),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 3, 5)
        )
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, return_both=False, return_features_only=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x.mean(dim=1))
        features = x
        
        if return_features_only:
            return features
        if return_both:
            logits = self.classifier_head(features)
            return features, logits
        else:
            return self.classifier_head(features)

class MedicalRAG(nn.Module):
    """Retrieval-Augmented Generation for medical report generation."""
    def __init__(self, vision_encoder=None, sparse_rate=0.4):
        super().__init__()
        self.vision_encoder = vision_encoder if vision_encoder is not None else MedicalViT(sparse_rate=sparse_rate)
        self.language_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.text_encoder = AutoModel.from_pretrained("allenai/radbert")
        self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/radbert")
        
        # Projection layers
        self.context_proj = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.retrieval_index = None
        self.report_db = []
        
    def build_index(self, dataset_loader):
        """Builds FAISS index for report retrieval."""
        all_embeds = []
        device = next(self.text_encoder.parameters()).device
        
        for batch in dataset_loader:
            reports = batch['report']
            inputs = self.text_tokenizer(
                reports,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                embeds = outputs.last_hidden_state.mean(dim=1)
                all_embeds.append(embeds.cpu().numpy())
                
        all_embeds = np.concatenate(all_embeds, axis=0)
        d = all_embeds.shape[1]
        
        self.retrieval_index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(all_embeds)
        self.retrieval_index.add(all_embeds)
        
    def retrieve_context(self, img_embed, k=3):
        """Retrieves relevant reports based on image embeddings."""
        if self.retrieval_index is None or len(self.report_db) == 0:
            return [""] * img_embed.size(0)
            
        img_embed_norm = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-8)
        scores, indices = self.retrieval_index.search(
            img_embed_norm.cpu().numpy(),
            k
        )
        
        context_reports = []
        for idx in indices:
            context = " ".join([self.report_db[i] for i in idx])
            context_reports.append(context)
            
        return context_reports
    
    def forward(self, images, target_reports=None, precomputed_features=None):
        """Forward pass with optional precomputed features."""
        if precomputed_features is None:
            img_features = self.vision_encoder(images, return_features_only=True)
        else:
            img_features = precomputed_features

        context_reports = self.retrieve_context(img_features)
        context_tokens = self.tokenizer(
            context_reports,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(images.device)
        
        with torch.no_grad():
            context_outputs = self.language_model.model.encoder(**context_tokens)
            context_embeds = context_outputs.last_hidden_state.mean(dim=1)
            
        fused = self.context_proj(torch.cat([img_features, context_embeds], dim=-1))
        
        if target_reports is not None:
            target_tokens = self.tokenizer(
                target_reports,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(images.device)
            
            outputs = self.language_model(
                inputs_embeds=fused.unsqueeze(1),
                labels=target_tokens.input_ids
            )
            
            loss = outputs.loss
            gen_ids = self.language_model.generate(
                inputs_embeds=fused.unsqueeze(1),
                max_length=128,
                num_beams=4,
                temperature=0.9,
                no_repeat_ngram_size=3
            )
            
            generated_reports = [
                self.tokenizer.decode(g, skip_special_tokens=True)
                for g in gen_ids
            ]
            
            return {"loss": loss, "generated_reports": generated_reports}
        else:
            gen_ids = self.language_model.generate(
                inputs_embeds=fused.unsqueeze(1),
                max_length=128,
                num_beams=4,
                temperature=0.9,
                no_repeat_ngram_size=3
            )
            
            return [
                self.tokenizer.decode(g, skip_special_tokens=True)
                for g in gen_ids
            ]