import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
import clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision.transforms.functional import to_pil_image
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MedicalEvaluator:
    """
    A class to evaluate generated medical reports using a suite of metrics:
    BLEU, CLIP score, and CheXbert.
    """
    def __init__(self, device: torch.device):
        """
        Args:
            device (torch.device): The device to run the models on (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self._init_metrics()
        
    def _init_metrics(self):
        """Initializes and loads all the necessary models for evaluation."""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            self.chexbert_tokenizer = AutoTokenizer.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model = AutoModelForSequenceClassification.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model.to(self.device)
            
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise
            
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculates BLEU scores (1 to 4) for generated reports.

        Args:
            predictions (List[str]): A list of generated reports.
            references (List[str]): A list of ground-truth reports.

        Returns:
            Dict[str, float]: A dictionary containing BLEU-1 to BLEU-4 scores.
        """
        if not predictions or not references:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}

        try:
            pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
            ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]
            
            weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1/3, 1/3, 1/3, 0), (0.25, 0.25, 0.25, 0.25)]
            scores = [corpus_bleu(ref_tokens, pred_tokens, w) for w in weights]
            
            return {f'bleu{i+1}': score for i, score in enumerate(scores)}
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return {}

    def calculate_clip_score(self, images: List[torch.Tensor], texts: List[str]) -> float:
        """
        Calculates the CLIP score for image-text alignment.

        Args:
            images (List[torch.Tensor]): A list of image tensors.
            texts (List[str]): A list of corresponding generated texts.

        Returns:
            float: The average CLIP similarity score.
        """
        if not images or not texts:
            return 0.0
            
        try:
            pil_images = [to_pil_image(img) for batch in images for img in batch]
            processed_images = torch.stack([self.clip_preprocess(p) for p in pil_images]).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(processed_images)
                text_tokens = clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).diagonal()
                return similarity.mean().item()
        except Exception as e:
            logger.error(f"CLIP score calculation failed: {e}")
            return 0.0

    def calculate_chexbert_score(self, predictions: List[str]) -> Dict[str, float]:
        """
        Calculates CheXbert scores for medical accuracy of the generated reports.

        Args:
            predictions (List[str]): A list of generated reports.

        Returns:
            Dict[str, float]: A dictionary with the mean CheXbert score.
        """
        if not predictions:
            return {'chexbert_mean': 0.0}
            
        try:
            with torch.no_grad():
                inputs = self.chexbert_tokenizer(
                    predictions, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                
                scores = torch.sigmoid(self.chexbert_model(**inputs).logits)
                return {'chexbert_mean': scores.mean().item()}
        except Exception as e:
            logger.error(f"CheXbert score calculation failed: {e}")
            return {}

    def evaluate(self, predictions: List[str], references: List[str], images: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Runs a comprehensive evaluation using all configured metrics.

        Args:
            predictions (List[str]): The generated reports.
            references (List[str]): The ground-truth reports.
            images (List[torch.Tensor]): The input images.
            
        Returns:
            Dict[str, Any]: A dictionary containing the results from all metrics.
        """
        return {
            'bleu': self.calculate_bleu(predictions, references),
            'clip_score': self.calculate_clip_score(images, predictions),
            'chexbert': self.calculate_chexbert_score(predictions)
        }