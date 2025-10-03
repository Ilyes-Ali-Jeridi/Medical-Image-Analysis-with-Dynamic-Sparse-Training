#evaluator.py
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
import clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class MedicalEvaluator:
    """
    A comprehensive evaluator for medical report generation systems.

    This class integrates multiple metrics to assess the quality of generated
    medical reports against reference texts and corresponding images. It calculates
    BLEU scores for textual similarity, CLIP scores for image-text alignment,
    and CheXbert scores for clinical accuracy based on common radiological findings.

    Attributes:
        device (torch.device): The PyTorch device (e.g., 'cuda', 'cpu') on which to run
                               the evaluation models (CLIP, CheXbert).
        clip_model: The loaded CLIP model.
        clip_preprocess: The preprocessing transform associated with the CLIP model.
        chexbert_tokenizer: The tokenizer for the CheXbert model.
        chexbert_model: The loaded CheXbert model.
    """
    def __init__(self, device: torch.device):
        """
        Initializes the MedicalEvaluator.

        Args:
            device (torch.device): The device to use for loading evaluation models.
        
        Raises:
            Various exceptions from underlying libraries if model loading/downloading fails
            (e.g., FileNotFoundError, ConnectionError, ImportError).
        """
        self.device = device
        logger.info(f"MedicalEvaluator initialized with device: {self.device}")
        self._init_metrics() # Load and prepare all metric computation models
        
    def _init_metrics(self):
        """
        Initializes and loads all necessary models and data for metric calculation.
        
        This includes loading the CLIP model, CheXbert model and tokenizer,
        and ensuring NLTK 'punkt' data is available for BLEU score calculation.
        Logs success or failure for each initialization step.

        Raises:
            FileNotFoundError: If a required model file is not found.
            ImportError: If a necessary library (e.g., clip, transformers, nltk) is not installed.
            ConnectionError: If there's an issue downloading models or NLTK data.
            Exception: For other unexpected errors during initialization.
        """
        try:
            # Initialize CLIP model and its associated preprocessor
            logger.info("Initializing CLIP model (ViT-B/32)...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model initialized successfully.")

            logger.info("Initializing CheXbert model and tokenizer...")
            self.chexbert_tokenizer = AutoTokenizer.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model = AutoModelForSequenceClassification.from_pretrained("stanfordaimi/CheXbert")
            self.chexbert_model.to(self.device)
            logger.info("CheXbert model and tokenizer initialized successfully.")
            
            # NLTK 'punkt' data download is expected to be handled globally, e.g., in main.py.
            # Verify its availability for tokenization.
            try:
                nltk.data.find('tokenizers/punkt')
                logger.info("NLTK 'punkt' tokenizer data found.")
            except LookupError:
                logger.error("NLTK 'punkt' tokenizer data not found. BLEU score calculation will likely fail. "
                             "Ensure `nltk.download('punkt')` is called at application startup.")
                # Depending on strictness, could raise an error here.
                # For now, logging error and relying on calculate_bleu to handle nltk failures.
            
        except FileNotFoundError as fnf_error:
            logger.error(f"Failed to initialize metrics: A model file for CLIP or CheXbert was not found. {fnf_error}", exc_info=True)
            raise
        except ImportError as imp_error:
             logger.error(f"Failed to initialize metrics: An import error occurred, possibly a missing dependency (e.g., 'clip', 'transformers', 'nltk'). {imp_error}", exc_info=True)
             raise
        except ConnectionError as conn_error: # More specific than generic Exception for network issues
            logger.error(f"Failed to initialize metrics: A connection error occurred, likely during model/data download. {conn_error}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other unexpected errors (e.g., from underlying libraries)
            logger.error(f"An unexpected error occurred during metrics initialization: {e}", exc_info=True)
            raise
            
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Optional[Dict[str, float]]:
        """
        Calculates BLEU scores (BLEU-1 to BLEU-4) for generated reports.

        Args:
            predictions (List[str]): A list of generated report strings.
            references (List[str]): A list of reference report strings, corresponding to predictions.

        Returns:
            Optional[Dict[str, float]]: A dictionary containing BLEU-1 to BLEU-4 scores,
            or None if calculation fails (e.g., empty inputs).
        """
        if not predictions or not references:
            logger.warning("BLEU calculation skipped: Predictions or references list is empty.")
            return None
        if len(predictions) != len(references):
            logger.warning(f"BLEU calculation: Mismatch in lengths of predictions ({len(predictions)}) and references ({len(references)}). Proceeding with shorter length.")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        try:
            # Tokenize predictions and references. NLTK expects references to be a list of lists of tokens.
            pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
            ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references] # Each reference is a list containing one tokenized version
            
            # Define weights for BLEU-1 to BLEU-4
            bleu_weights = {
                'bleu1': (1.0, 0.0, 0.0, 0.0),
                'bleu2': (0.5, 0.5, 0.0, 0.0),
                'bleu3': (0.333, 0.333, 0.333, 0.0), # Approximate 1/3
                'bleu4': (0.25, 0.25, 0.25, 0.25)
            }
            bleu_scores_dict = {}
            
            for name, weight in bleu_weights.items():
                # corpus_bleu calculates BLEU score for a corpus of sentences
                score = corpus_bleu(ref_tokens, pred_tokens, weights=weight)
                bleu_scores_dict[name] = score
            
            logger.debug(f"Calculated BLEU scores: {bleu_scores_dict}")
            return bleu_scores_dict
        except ValueError as ve: # Can occur if, e.g., all sentences are extremely short
            logger.error(f"BLEU calculation failed due to a ValueError (e.g., empty token lists, short sentences): {ve}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during BLEU calculation: {e}", exc_info=True)
            return None
            
    def calculate_clip_score(self, images: List[Image.Image], texts: List[str]) -> Optional[float]:
        """
        Calculates the average CLIP score for image-text alignment.

        Args:
            images (List[Image.Image]): A list of PIL Images.
            texts (List[str]): A list of text strings corresponding to the images.

        Returns:
            Optional[float]: The average CLIP score, or None if calculation fails.
        """
        if not images or not texts:
            logger.warning("CLIP score calculation skipped: Images or texts list is empty.")
            return None
        if len(images) != len(texts):
            logger.warning(f"CLIP score calculation: Mismatch in lengths of images ({len(images)}) and texts ({len(texts)}). Results may be misleading.")
            # Decide on handling: either error, or proceed with min_len. For now, proceed.
            min_len = min(len(images), len(texts))
            images = images[:min_len]
            texts = texts[:min_len]
            if not images: # If min_len was 0
                logger.warning("CLIP score calculation skipped after length adjustment: No pairs to evaluate.")
                return None

        try:
            with torch.no_grad(): # Ensure no gradients are computed
                # Preprocess all images and stack them into a batch tensor
                processed_images = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
                # Encode images to get image features
                image_features = self.clip_model.encode_image(processed_images)
                
                # Tokenize and encode texts to get text features
                # CLIP's tokenize function handles batching of texts.
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device) # Truncate ensures texts fit context length
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features for cosine similarity calculation
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate dot product similarity (cosine similarity due to normalization)
                # We are interested in the diagonal of the similarity matrix, which corresponds to image_i vs text_i pairs.
                similarity = (image_features @ text_features.T).diagonal()
                clip_score = similarity.mean().item() # Average similarity for the batch
                
                logger.debug(f"Calculated average CLIP score: {clip_score} for {len(images)} pairs.")
                return clip_score
        except RuntimeError as rt_e:
            logger.error(f"CLIP score calculation failed due to a runtime error (e.g., OOM, device mismatch): {rt_e}", exc_info=True)
            return None
        except IndexError as idx_e: # If .diagonal() fails (e.g. if image_features or text_features is unexpectedly empty)
            logger.error(f"CLIP score calculation failed due to an IndexError (likely tensor shape mismatch or empty features): {idx_e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during CLIP score calculation: {e}", exc_info=True)
            return None
            
    def calculate_chexbert_score(self, predictions: List[str]) -> Optional[np.ndarray]:
        """
        Calculates CheXbert scores for medical accuracy of predicted reports.

        Args:
            predictions (List[str]): A list of generated report strings.

        Returns:
            Optional[np.ndarray]: A NumPy array of scores (average per prediction across conditions),
            or None if calculation fails.
        """
        if not predictions:
            logger.warning("CheXbert score calculation skipped: Predictions list is empty.")
            return None

        try:
            with torch.no_grad(): # Ensure no gradients are computed
                # Tokenize predictions. CheXbert tokenizer handles batching.
                inputs = self.chexbert_tokenizer(
                    predictions,
                    padding=True,       # Pad to max length in batch
                    truncation=True,    # Truncate reports longer than model's max input size
                    return_tensors="pt" # Return PyTorch tensors
                ).to(self.device)       # Move tensors to the specified device
                
                # Get model outputs (logits for different radiological findings)
                outputs = self.chexbert_model(**inputs)
                # Apply sigmoid to logits to get probabilities (or pseudo-probabilities)
                scores = torch.sigmoid(outputs.logits)
                
                # Average scores across all medical conditions for each prediction
                # This gives a single summary score per report.
                chexbert_scores_avg = scores.mean(dim=1).cpu().numpy() # Move to CPU and convert to NumPy array
                
                logger.debug(f"Calculated CheXbert scores (mean per prediction): {chexbert_scores_avg}")
                return chexbert_scores_avg
        except RuntimeError as rt_e: # E.g., OOM, issues with HuggingFace model forward pass
            logger.error(f"CheXbert score calculation failed due to a runtime error: {rt_e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during CheXbert score calculation: {e}", exc_info=True)
            return None
            
    def evaluate(self, predictions: List[str], references: List[str], images: List[Image.Image]) -> Dict[str, Any]:
        """
        Performs a comprehensive evaluation using all configured metrics.

        Args:
            predictions (List[str]): List of generated report strings.
            references (List[str]): List of reference report strings.
            images (List[Image.Image]): List of PIL Images corresponding to the reports.

        Returns:
            Dict[str, Any]: A dictionary containing the results from all metrics.
                            Keys are metric names (e.g., 'bleu', 'clip', 'chexbert'),
                            and values are their respective scores. If a metric fails,
                            its value might be None or an error indicator.
        """
        logger.info(f"Starting comprehensive evaluation for {len(predictions)} predictions...")
        results = {} # Dictionary to store all evaluation results
        
        # Calculate BLEU scores
        logger.info("Calculating BLEU scores...")
        bleu_scores = self.calculate_bleu(predictions, references)
        if bleu_scores is None:
            logger.warning("BLEU score calculation failed. Results will not include 'bleu'.")
        results['bleu'] = bleu_scores

        # Calculate CLIP score
        logger.info("Calculating CLIP score...")
        clip_score = self.calculate_clip_score(images, predictions)
        if clip_score is None:
            logger.warning("CLIP score calculation failed. Results will not include 'clip'.")
        results['clip'] = clip_score
            
        # Calculate CheXbert scores
        logger.info("Calculating CheXbert scores...")
        chexbert_scores = self.calculate_chexbert_score(predictions)
        if chexbert_scores is None:
            logger.warning("CheXbert score calculation failed. Results will not include 'chexbert'.")
        results['chexbert'] = chexbert_scores # This might be an array of scores
        
        logger.info(f"Comprehensive evaluation completed. Results: {results}")
        return results