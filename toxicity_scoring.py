"""
Toxicity Scoring Module for CDRC Framework
Handles training biased Detoxify models and computing toxicity scores
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import logging
from detoxify import Detoxify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JigsawDataset(Dataset):
    """Custom dataset for Jigsaw toxic comment data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class ToxicityScoringModule:
    def __init__(self, base_model="unitary/toxic-bert", device="cuda"):
        """
        Initialize toxicity scoring module
        
        Args:
            base_model: Base model for toxicity detection
            device: Device to run models on
        """
        self.base_model = base_model
        self.device = device
        
        # Load base Detoxify model for comparison
        self.detoxify_original = Detoxify('original', device=device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    def load_jigsaw_data(self, data_path="./jigsaw_data/train.csv"):
        """
        Load Jigsaw toxic comment classification data
        
        Args:
            data_path: Path to Jigsaw CSV file
            
        Returns:
            DataFrame with text and toxicity labels
        """
        logger.info("Loading Jigsaw dataset...")
        
        # Download if not exists
        if not os.path.exists(data_path):
            logger.info("Downloading Jigsaw dataset...")
            # You would need to implement download logic or use kaggle API
            # For now, assume the file exists
            raise FileNotFoundError(f"Please download Jigsaw data to {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Extract text and toxicity scores
        texts = df['comment_text'].values
        
        # Combine all toxicity labels into single score
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        toxicity_scores = df[toxic_cols].max(axis=1).values
        
        return texts, toxicity_scores
    
    def create_biased_subset(self, texts, labels, bias_percentage=0.15, random_state=42):
        """
        Create biased subset of data by manipulating label distribution
        
        Args:
            texts: Input texts
            labels: Toxicity labels (0 or 1)
            bias_percentage: Percentage of toxic samples to flip to non-toxic
            random_state: Random seed
            
        Returns:
            Biased texts and labels
        """
        np.random.seed(random_state)
        
        # Convert to binary labels
        binary_labels = (labels > 0.5).astype(int)
        
        # Find toxic samples
        toxic_indices = np.where(binary_labels == 1)[0]
        
        # Randomly select samples to flip
        num_to_flip = int(len(toxic_indices) * bias_percentage)
        flip_indices = np.random.choice(toxic_indices, num_to_flip, replace=False)
        
        # Create biased labels
        biased_labels = binary_labels.copy()
        biased_labels[flip_indices] = 0
        
        logger.info(f"Created biased dataset: flipped {num_to_flip} toxic samples")
        logger.info(f"Original toxic ratio: {binary_labels.mean():.3f}")
        logger.info(f"Biased toxic ratio: {biased_labels.mean():.3f}")
        
        return texts, biased_labels
    
    def train_biased_model(self, texts, labels, model_name, output_dir, 
                          epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train a biased toxicity detection model
        
        Args:
            texts: Training texts
            labels: Training labels
            model_name: Name for the model (e.g., "detoxify_15")
            output_dir: Directory to save trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Path to saved model
        """
        logger.info(f"Training biased model: {model_name}")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        
        # Create datasets
        train_dataset = JigsawDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = JigsawDataset(val_texts, val_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, 
            num_labels=1,
            problem_type="regression"
        ).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    preds = torch.sigmoid(outputs.logits).squeeze()
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            val_corr, _ = spearmanr(val_preds, val_true)
            logger.info(f"Validation Spearman correlation: {val_corr:.4f}")
            
            model.train()
        
        # Save model
        model_path = os.path.join(output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_biased_model(self, model_path):
        """
        Load a trained biased model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model and tokenizer
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    
    def score_texts_with_model(self, texts, model, tokenizer, batch_size=32):
        """
        Score texts using a specific model
        
        Args:
            texts: List of texts to score
            model: Trained model
            tokenizer: Model tokenizer
            batch_size: Batch size for inference
            
        Returns:
            Array of toxicity scores
        """
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**encodings)
                batch_scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
                
            if len(batch_texts) == 1:
                batch_scores = [batch_scores.item()]
            
            scores.extend(batch_scores)
        
        return np.array(scores)
    
    def compute_human_machine_alignment(self, texts, human_scores, machine_scores):
        """
        Compute Spearman correlation between human and machine scores
        
        Args:
            texts: Input texts
            human_scores: Human-annotated toxicity scores
            machine_scores: Machine-predicted toxicity scores
            
        Returns:
            Spearman correlation coefficient and p-value
        """
        # Remove any NaN values
        mask = ~(np.isnan(human_scores) | np.isnan(machine_scores))
        human_scores_clean = human_scores[mask]
        machine_scores_clean = machine_scores[mask]
        
        corr, p_value = spearmanr(human_scores_clean, machine_scores_clean)
        
        logger.info(f"Human-Machine Alignment: {corr:.4f} (p={p_value:.4e})")
        return corr, p_value
    
    def train_all_biased_models(self, jigsaw_data_path, output_dir="./biased_models",
                               bias_percentages=[0.15, 0.30, 0.70]):
        """
        Train all biased models with different bias levels
        
        Args:
            jigsaw_data_path: Path to Jigsaw dataset
            output_dir: Directory to save models
            bias_percentages: List of bias percentages to train
            
        Returns:
            Dictionary of model paths
        """
        # Load original data
        texts, labels = self.load_jigsaw_data(jigsaw_data_path)
        
        model_paths = {}
        
        for bias_pct in bias_percentages:
            # Create biased dataset
            biased_texts, biased_labels = self.create_biased_subset(
                texts, labels, bias_percentage=bias_pct
            )
            
            # Train model
            model_name = f"detoxify_{int(bias_pct*100)}"
            model_path = self.train_biased_model(
                biased_texts, biased_labels, 
                model_name, output_dir
            )
            
            model_paths[bias_pct] = model_path
        
        return model_paths
    
    def score_candidate_sets(self, x_cal, y_cal, model_paths, human_scores_path=None):
        """
        Score all candidate sets with biased models
        
        Args:
            x_cal: Calibration features dictionary
            y_cal: Calibration candidate sets
            model_paths: Dictionary of trained model paths
            human_scores_path: Path to human toxicity scores (if available)
            
        Returns:
            Updated x_cal with toxicity scores
        """
        # Load models
        models = {}
        for bias_pct, path in model_paths.items():
            model, tokenizer = self.load_biased_model(path)
            models[bias_pct] = (model, tokenizer)
        
        # Use the 15% biased model as default
        default_model, default_tokenizer = models[0.15]
        
        # Score all texts
        for idx, data in tqdm(x_cal.items(), desc="Scoring candidate sets"):
            texts = data['pred']
            
            # Score with biased model
            machine_scores = self.score_texts_with_model(
                texts, default_model, default_tokenizer
            )
            
            # Store scores
            x_cal[idx]['detoxify_ft'] = [machine_scores]
            
            # Add placeholder for human scores (to be filled from actual annotations)
            if human_scores_path:
                # Load human scores from file
                with open(human_scores_path, 'rb') as f:
                    human_data = pickle.load(f)
                if idx in human_data:
                    x_cal[idx]['detoxify_human'] = human_data[idx]
            else:
                # Simulate human scores for testing
                x_cal[idx]['detoxify_human'] = machine_scores + np.random.normal(0, 0.1, len(machine_scores))
        
        return x_cal


# Example usage
if __name__ == "__main__":
    # Initialize scoring module
    scorer = ToxicityScoringModule()
    
    # Train biased models
    model_paths = scorer.train_all_biased_models(
        jigsaw_data_path="./jigsaw_data/train.csv",
        bias_percentages=[0.15, 0.30, 0.70]
    )
    
    # Load calibration data
    with open("./processed_data/processed_data_complete.pkl", 'rb') as f:
        processed_data = pickle.load(f)
    
    # Prepare x_cal and y_cal
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    x_cal, y_cal = preprocessor.prepare_calibration_data("./processed_data/processed_data_complete.pkl")
    
    # Score candidate sets
    x_cal_scored = scorer.score_candidate_sets(x_cal, y_cal, model_paths)
    
    # Save scored data
    with open("./processed_data/x_cal_scored.pkl", 'wb') as f:
        pickle.dump(x_cal_scored, f)