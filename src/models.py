import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModel, AutoTokenizer
import logging
import faiss
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DynamicSparseLayer(nn.Module):
    """
    A dynamically sparse layer that prunes weights based on magnitude.
    This layer wraps a standard nn.Module and applies a mask to its weights,
    which is updated periodically based on gradient information.
    """
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
        """Applies the masked weights to the input tensor."""
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
            logger.error(f"Forward pass failed in DynamicSparseLayer: {str(e)}")
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
    """
    Vision Transformer (ViT) tailored for medical imaging tasks.
    It uses dynamic sparsity and is configurable via a central config object.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.patch_embed = DynamicSparseLayer(
            nn.Conv2d(1, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size),
            sparsity=config.sparse_rate
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=8,  # Typically embed_dim // 64
                dim_feedforward=config.embed_dim * 4,
                dropout=0.1,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier_head = nn.Sequential(
            DynamicSparseLayer(nn.Linear(config.embed_dim, config.embed_dim // 3), sparsity=config.sparse_rate),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 3, 5) # 5 classes for classification
        )
        
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor, return_both: bool = False, return_features_only: bool = False):
        """
        Forward pass for the MedicalViT.
        Can return classification logits, features, or both.
        """
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
    """
    Retrieval-Augmented Generation (RAG) model for medical report generation.
    It combines a vision encoder with a language model and a retrieval mechanism.
    """
    def __init__(self, config: Any, vision_encoder: nn.Module = None):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder or MedicalViT(config)

        self.language_model = BartForConditionalGeneration.from_pretrained(config.language_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(config.language_model_name)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        
        self.context_proj = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.retrieval_index = None
        self.report_db = []
        
    def build_index(self, dataset_loader):
        """Builds a FAISS index from the reports in the dataset for fast retrieval."""
        all_embeds = []
        device = next(self.text_encoder.parameters()).device
        
        for batch in dataset_loader:
            reports = batch['report']
            inputs = self.text_tokenizer(
                reports, padding=True, truncation=True, return_tensors="pt", max_length=512
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
        
    def retrieve_context(self, img_embed: torch.Tensor, k: int = 3) -> list:
        """Retrieves k-nearest reports from the index based on image embeddings."""
        if self.retrieval_index is None or not self.report_db:
            return [""] * img_embed.size(0)
            
        img_embed_norm = F.normalize(img_embed, p=2, dim=-1)
        scores, indices = self.retrieval_index.search(img_embed_norm.cpu().numpy(), k)
        
        return [" ".join([self.report_db[i] for i in idx]) for idx in indices]
    
    def forward(self, images: torch.Tensor, target_reports: list = None, precomputed_features: torch.Tensor = None) -> Dict[str, Any]:
        """
        Forward pass for the RAG model.
        Generates reports and calculates loss if target reports are provided.
        """
        if precomputed_features is None:
            img_features = self.vision_encoder(images, return_features_only=True)
        else:
            img_features = precomputed_features

        context_reports = self.retrieve_context(img_features)
        context_tokens = self.tokenizer(
            context_reports, return_tensors="pt", truncation=True, max_length=128, padding=True
        ).to(images.device)
        
        with torch.no_grad():
            context_outputs = self.language_model.model.encoder(**context_tokens)
            context_embeds = context_outputs.last_hidden_state.mean(dim=1)
            
        fused = self.context_proj(torch.cat([img_features, context_embeds], dim=-1))
        
        if target_reports is not None:
            target_tokens = self.tokenizer(
                target_reports, return_tensors="pt", truncation=True, padding="max_length", max_length=128
            ).to(images.device)
            
            outputs = self.language_model(
                inputs_embeds=fused.unsqueeze(1), labels=target_tokens.input_ids
            )
            
            gen_ids = self.language_model.generate(
                inputs_embeds=fused.unsqueeze(1), max_length=128, num_beams=4, temperature=0.9, no_repeat_ngram_size=3
            )
            
            generated_reports = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            return {"loss": outputs.loss, "generated_reports": generated_reports}
        else:
            gen_ids = self.language_model.generate(
                inputs_embeds=fused.unsqueeze(1), max_length=128, num_beams=4, temperature=0.9, no_repeat_ngram_size=3
            )
            
            return {"generated_reports": self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)}