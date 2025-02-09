import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
import clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class MedicalEvaluator:
    """Evaluator for medical reports using BLEU, CLIP, and CheXbert metrics."""
    def __init__(self, device):
        self.device = device
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize all evaluation metrics."""
        try:
            # Initialize CLIP
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Initialize CheXbert
            self.chexbert_tokenizer = AutoTokenizer.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model = AutoModelForSequenceClassification.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model.to(self.device)
            
            # Download NLTK data for BLEU
            nltk.download('punkt')
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {str(e)}")
            raise
            
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score for generated reports."""
        try:
            # Tokenize predictions and references
            pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
            ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]
            
            # Calculate BLEU-1 to BLEU-4
            weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
            bleu_scores = []
            
            for weight in weights:
                score = corpus_bleu(ref_tokens, pred_tokens, weights=weight)
                bleu_scores.append(score)
                
            return {
                'bleu1': bleu_scores[0],
                'bleu2': bleu_scores[1],
                'bleu3': bleu_scores[2],
                'bleu4': bleu_scores[3]
            }
        except Exception as e:
            logger.error(f"BLEU calculation failed: {str(e)}")
            return None
            
    def calculate_clip_score(self, images, texts):
        """Calculate CLIP score for image-text alignment."""
        try:
            with torch.no_grad():
                # Preprocess images and encode
                processed_images = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
                image_features = self.clip_model.encode_image(processed_images)
                
                # Encode text
                text_tokens = clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (image_features @ text_features.T).diagonal()
                return similarity.mean().item()
        except Exception as e:
            logger.error(f"CLIP score calculation failed: {str(e)}")
            return None
            
    def calculate_chexbert_score(self, predictions):
        """Calculate CheXbert scores for medical accuracy."""
        try:
            with torch.no_grad():
                inputs = self.chexbert_tokenizer(
                    predictions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.chexbert_model(**inputs)
                scores = torch.sigmoid(outputs.logits)
                
                # Average across all medical conditions
                return scores.mean(dim=1).cpu().numpy()
        except Exception as e:
            logger.error(f"CheXbert score calculation failed: {str(e)}")
            return None
            
    def evaluate(self, predictions, references, images):
        """Comprehensive evaluation using all metrics."""
        results = {
            'bleu': self.calculate_bleu(predictions, references),
            'clip': self.calculate_clip_score(images, predictions),
            'chexbert': self.calculate_chexbert_score(predictions)
        }
        
        return results