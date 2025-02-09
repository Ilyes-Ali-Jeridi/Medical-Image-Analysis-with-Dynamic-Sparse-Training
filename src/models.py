#models.py
# Update the MedicalRAG class fusion mechanism
class MedicalRAG(nn.Module):
    def __init__(self, vision_encoder=None, sparse_rate=0.4):
        super().__init__()
        self.vision_encoder = vision_encoder if vision_encoder is not None else MedicalViT(sparse_rate=sparse_rate)
        self.language_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.text_encoder = AutoModel.from_pretrained("allenai/radbert")
        self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/radbert")
        
        # Enhanced fusion mechanism as per paper
        self.img_proj = nn.Linear(768, 768)
        self.text_proj = nn.Linear(768, 768)
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1)
        )
        
    def fuse_embeddings(self, img_features, text_features):
        """Enhanced fusion mechanism as described in paper section 3."""
        # Project features to common space
        img_proj = self.img_proj(img_features)
        text_proj = self.text_proj(text_features)
        
        # Attention-based fusion
        attention = torch.matmul(img_proj, text_proj.transpose(-2, -1))
        attention = F.softmax(attention / torch.sqrt(torch.tensor(768.0)), dim=-1)
        
        # Combine features
        fused = torch.cat([
            img_proj,
            torch.matmul(attention, text_proj)
        ], dim=-1)
        
        # Final fusion through MLP
        return self.fusion_layer(fused)
        
    def forward(self, images, target_reports=None, precomputed_features=None):
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
            
        # Use enhanced fusion
        fused = self.fuse_embeddings(img_features, context_embeds)
        
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