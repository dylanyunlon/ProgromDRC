"""
Script to download Jigsaw Toxic Comment Classification dataset
Note: You need to have kaggle API credentials set up
"""

import os
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_jigsaw_data():
    """
    Download Jigsaw dataset using Kaggle API
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get API credentials from https://www.kaggle.com/account
    3. Place kaggle.json in ~/.kaggle/
    """
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    
    # Create directory
    os.makedirs('./jigsaw_data', exist_ok=True)
    
    try:
        # Download dataset
        logger.info("Downloading Jigsaw dataset from Kaggle...")
        kaggle.api.competition_download_file(
            'jigsaw-toxic-comment-classification-challenge',
            'train.csv.zip',
            path='./jigsaw_data'
        )
        
        # Extract zip file
        logger.info("Extracting dataset...")
        with zipfile.ZipFile('./jigsaw_data/train.csv.zip', 'r') as zip_ref:
            zip_ref.extractall('./jigsaw_data')
        
        # Remove zip file
        os.remove('./jigsaw_data/train.csv.zip')
        
        logger.info("✓ Jigsaw dataset downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("\nAlternative: Manual download")
        logger.info("1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
        logger.info("2. Download 'train.csv.zip'")
        logger.info("3. Extract and place 'train.csv' in ./jigsaw_data/")
        return False


def create_sample_jigsaw_data():
    """Create a sample Jigsaw dataset for testing"""
    import pandas as pd
    import numpy as np
    
    logger.info("Creating sample Jigsaw dataset for testing...")
    
    # Create sample data
    n_samples = 1000
    texts = [
        "This is a normal comment about the weather.",
        "I really hate this stupid thing!",
        "You are an idiot and should die!",
        "Great article, thanks for sharing.",
        "This is absolutely terrible and disgusting.",
        "I love how you explained this topic.",
        "Go kill yourself, nobody likes you!",
        "Interesting perspective on the issue.",
        "You're so dumb, I can't believe it.",
        "Thanks for the helpful information."
    ] * (n_samples // 10)
    
    # Create toxicity labels
    toxic_patterns = ['hate', 'stupid', 'idiot', 'die', 'kill', 'dumb', 'terrible', 'disgusting']
    
    data = {
        'id': [f'id_{i}' for i in range(n_samples)],
        'comment_text': texts[:n_samples]
    }
    
    # Add toxicity labels
    for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        data[col] = []
        for text in texts[:n_samples]:
            # Simple rule-based labeling for demo
            if col == 'toxic':
                label = 1 if any(word in text.lower() for word in toxic_patterns) else 0
            elif col == 'threat':
                label = 1 if any(word in text.lower() for word in ['die', 'kill']) else 0
            elif col == 'insult':
                label = 1 if any(word in text.lower() for word in ['idiot', 'dumb', 'stupid']) else 0
            else:
                label = np.random.choice([0, 1], p=[0.95, 0.05])
            
            data[col].append(label)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('./jigsaw_data', exist_ok=True)
    df.to_csv('./jigsaw_data/train.csv', index=False)
    
    logger.info(f"✓ Created sample dataset with {n_samples} examples")
    logger.info(f"  Toxic ratio: {df['toxic'].mean():.2%}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        # Create sample dataset
        create_sample_jigsaw_data()
    else:
        # Try to download real dataset
        success = download_jigsaw_data()
        if not success:
            logger.info("\nCreating sample dataset instead...")
            create_sample_jigsaw_data()
