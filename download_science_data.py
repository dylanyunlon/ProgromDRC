"""
Science Dataset Download and Preparation Script
Downloads and prepares scientific text datasets for CDRC framework
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceDataDownloader:
    def __init__(self, cache_dir="./science_data", output_dir="./processed_science_data"):
        """
        Initialize science data downloader
        
        Args:
            cache_dir: Directory to cache raw datasets
            output_dir: Directory to save processed data
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Scientific connectives for splitting sentences
        self.connectives = [
            "therefore", "thus", "hence", "consequently", "as a result",
            "this leads to", "which results in", "causing", "due to",
            "because of this", "accordingly", "so", "thereby"
        ]
        
        # Field indicators
        self.chemistry_keywords = [
            "reaction", "compound", "molecule", "element", "chemical",
            "synthesis", "catalyst", "acid", "base", "organic", "inorganic",
            "bond", "electron", "ion", "solution", "concentration"
        ]
        
        self.physics_keywords = [
            "force", "energy", "momentum", "velocity", "acceleration",
            "field", "wave", "particle", "quantum", "relativity",
            "thermodynamic", "electric", "magnetic", "gravity", "mass"
        ]
    
    def download_sciq_dataset(self) -> List[Dict]:
        """
        Download and process SciQ dataset
        
        Returns:
            List of processed prompts
        """
        logger.info("Downloading SciQ dataset...")
        dataset = load_dataset("sciq", split="train")
        
        prompts = []
        for item in tqdm(dataset, desc="Processing SciQ"):
            # Extract question and support text
            question = item['question']
            support = item['support']
            
            if support and len(support) > 50:  # Ensure meaningful support text
                # Try to split support text at natural break points
                prompt_data = self._create_prompt_from_text(support, question)
                if prompt_data:
                    prompt_data['source'] = 'sciq'
                    prompt_data['field'] = self._classify_field(support)
                    prompts.append(prompt_data)
        
        logger.info(f"Processed {len(prompts)} prompts from SciQ")
        return prompts
    
    def download_s2orc_sample(self, num_papers=1000) -> List[Dict]:
        """
        Download sample from S2ORC dataset
        
        Args:
            num_papers: Number of papers to sample
            
        Returns:
            List of processed prompts
        """
        logger.info("Downloading S2ORC sample...")
        
        # Use streaming to avoid downloading entire dataset
        dataset = load_dataset(
            "json", 
            data_files="/data/jiacheng/dylan/aaai/ProgromDRC/s2orc/data/metadata/sample.jsonl",
            split="train",
            streaming=True
        )
        
        prompts = []
        papers_processed = 0
        
        for paper in dataset:
            if papers_processed >= num_papers:
                break
                
            # Filter for chemistry/physics papers
            if paper.get('abstract') and paper.get('mag_field_of_study'):
                fields = paper['mag_field_of_study']
                if any(field in ['Chemistry', 'Physics'] for field in fields):
                    abstract = paper['abstract']
                    
                    # Create prompts from abstract
                    abstract_prompts = self._extract_prompts_from_abstract(abstract)
                    for prompt_data in abstract_prompts:
                        prompt_data['source'] = 's2orc'
                        prompt_data['paper_id'] = paper.get('paper_id', '')
                        prompt_data['field'] = 'chemistry' if 'Chemistry' in fields else 'physics'
                        prompts.append(prompt_data)
                    
                    papers_processed += 1
        
        logger.info(f"Processed {len(prompts)} prompts from {papers_processed} S2ORC papers")
        return prompts
    
    def download_arxiv_abstracts(self, num_papers=500, categories=['physics', 'chem-ph']) -> List[Dict]:
        """
        Download abstracts from arXiv API
        
        Args:
            num_papers: Number of papers to download
            categories: arXiv categories to search
            
        Returns:
            List of processed prompts
        """
        logger.info("Downloading arXiv abstracts...")
        prompts = []
        
        for category in categories:
            # Construct API query
            base_url = 'http://export.arxiv.org/api/query?'
            query = f'cat:{category}&start=0&max_results={num_papers}'
            
            response = requests.get(base_url + query)
            
            if response.status_code == 200:
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Extract abstracts
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    abstract_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    if abstract_elem is not None:
                        abstract = abstract_elem.text.strip()
                        
                        # Create prompts from abstract
                        abstract_prompts = self._extract_prompts_from_abstract(abstract)
                        for prompt_data in abstract_prompts:
                            prompt_data['source'] = 'arxiv'
                            prompt_data['field'] = 'physics' if 'physics' in category else 'chemistry'
                            prompts.append(prompt_data)
        
        logger.info(f"Processed {len(prompts)} prompts from arXiv")
        return prompts
    
    def create_textbook_style_prompts(self) -> List[Dict]:
        """
        Create textbook-style scientific prompts
        
        Returns:
            List of manually crafted prompts
        """
        textbook_prompts = [
            # Chemistry prompts
            {
                "prompt": "When the temperature of a chemical reaction is increased by 10°C,",
                "field": "chemistry",
                "concept": "reaction kinetics",
                "difficulty": "intermediate"
            },
            {
                "prompt": "In an endothermic reaction, adding heat to the system will",
                "field": "chemistry",
                "concept": "Le Chatelier's principle",
                "difficulty": "intermediate"
            },
            {
                "prompt": "When a strong acid is diluted with water, the pH of the solution",
                "field": "chemistry",
                "concept": "acid-base chemistry",
                "difficulty": "basic"
            },
            {
                "prompt": "The addition of a catalyst to a chemical reaction",
                "field": "chemistry",
                "concept": "catalysis",
                "difficulty": "intermediate"
            },
            {
                "prompt": "When sodium metal is exposed to water,",
                "field": "chemistry",
                "concept": "alkali metal reactions",
                "difficulty": "basic"
            },
            
            # Physics prompts
            {
                "prompt": "According to Newton's third law, when object A exerts a force on object B,",
                "field": "physics",
                "concept": "Newton's laws",
                "difficulty": "basic"
            },
            {
                "prompt": "As an object approaches the speed of light, its relativistic mass",
                "field": "physics",
                "concept": "special relativity",
                "difficulty": "advanced"
            },
            {
                "prompt": "In a closed system with no external forces, the total momentum",
                "field": "physics",
                "concept": "conservation laws",
                "difficulty": "intermediate"
            },
            {
                "prompt": "When light passes from air into water, its speed",
                "field": "physics",
                "concept": "optics",
                "difficulty": "basic"
            },
            {
                "prompt": "According to the uncertainty principle, the more precisely we know a particle's position,",
                "field": "physics",
                "concept": "quantum mechanics",
                "difficulty": "advanced"
            }
        ]
        
        for prompt in textbook_prompts:
            prompt['source'] = 'textbook_manual'
        
        return textbook_prompts
    
    def _create_prompt_from_text(self, text: str, context: str = "") -> Dict:
        """
        Create a prompt by splitting text at scientific connectives
        
        Args:
            text: Source text
            context: Additional context (e.g., question)
            
        Returns:
            Prompt dictionary or None
        """
        text = text.strip()
        
        # Find connective words
        for connective in self.connectives:
            pattern = rf'\s+{connective}\s+'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if matches:
                # Use the first match
                match = matches[0]
                prompt = text[:match.start()].strip()
                continuation = text[match.start():].strip()
                
                if len(prompt) > 20 and len(continuation) > 20:
                    return {
                        'prompt': prompt + ',',
                        'reference_continuation': continuation,
                        'connective': connective,
                        'full_text': text,
                        'context': context
                    }
        
        # Try splitting at commas for cause-effect relationships
        if ', ' in text:
            parts = text.split(', ', 1)
            if len(parts[0]) > 30 and len(parts[1]) > 30:
                return {
                    'prompt': parts[0] + ',',
                    'reference_continuation': parts[1],
                    'connective': 'comma',
                    'full_text': text,
                    'context': context
                }
        
        return None
    
    def _extract_prompts_from_abstract(self, abstract: str) -> List[Dict]:
        """
        Extract multiple prompts from a scientific abstract
        
        Args:
            abstract: Paper abstract text
            
        Returns:
            List of prompt dictionaries
        """
        prompts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', abstract)
        
        for i, sentence in enumerate(sentences):
            # Try to create prompt from individual sentences
            prompt_data = self._create_prompt_from_text(sentence)
            if prompt_data:
                prompts.append(prompt_data)
            
            # Try combining consecutive sentences
            if i < len(sentences) - 1:
                combined = sentence + '. ' + sentences[i + 1]
                prompt_data = self._create_prompt_from_text(combined)
                if prompt_data:
                    prompts.append(prompt_data)
        
        return prompts
    
    def _classify_field(self, text: str) -> str:
        """
        Classify text as chemistry or physics based on keywords
        
        Args:
            text: Text to classify
            
        Returns:
            'chemistry', 'physics', or 'general'
        """
        text_lower = text.lower()
        
        chem_count = sum(1 for keyword in self.chemistry_keywords if keyword in text_lower)
        phys_count = sum(1 for keyword in self.physics_keywords if keyword in text_lower)
        
        if chem_count > phys_count:
            return 'chemistry'
        elif phys_count > chem_count:
            return 'physics'
        else:
            return 'general'
    
    def merge_and_save_datasets(self, datasets: List[List[Dict]], output_file: str) -> str:
        """
        Merge multiple datasets and save
        
        Args:
            datasets: List of prompt lists
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        # Merge all prompts
        all_prompts = []
        for dataset in datasets:
            all_prompts.extend(dataset)
        
        # Remove duplicates based on prompt text
        seen_prompts = set()
        unique_prompts = []
        
        for prompt in all_prompts:
            prompt_text = prompt['prompt']
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                unique_prompts.append(prompt)
        
        # Add indices
        for i, prompt in enumerate(unique_prompts):
            prompt['idx'] = i
        
        # Save to file
        output_path = os.path.join(self.output_dir, output_file)
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(unique_prompts, f)
        
        # Also save as JSON for readability
        json_path = output_path.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            json.dump(unique_prompts, f, indent=2)
        
        logger.info(f"Saved {len(unique_prompts)} unique prompts to {output_path}")
        
        # Print statistics
        fields = [p['field'] for p in unique_prompts]
        sources = [p['source'] for p in unique_prompts]
        
        logger.info("\nDataset Statistics:")
        logger.info(f"Total prompts: {len(unique_prompts)}")
        logger.info(f"Chemistry prompts: {fields.count('chemistry')}")
        logger.info(f"Physics prompts: {fields.count('physics')}")
        logger.info(f"General prompts: {fields.count('general')}")
        logger.info("\nSources:")
        for source in set(sources):
            logger.info(f"  {source}: {sources.count(source)}")
        
        return output_path


def main():
    """Main function to download and prepare all datasets"""
    downloader = ScienceDataDownloader()
    
    # Download datasets
    logger.info("Starting dataset download...")
    
    # 1. SciQ dataset
    sciq_prompts = downloader.download_sciq_dataset()
    
    # 2. S2ORC sample (adjust number based on needs)
    s2orc_prompts = downloader.download_s2orc_sample(num_papers=500)
    
    # 3. arXiv abstracts
    arxiv_prompts = downloader.download_arxiv_abstracts(num_papers=300)
    
    # 4. Manual textbook-style prompts
    textbook_prompts = downloader.create_textbook_style_prompts()
    
    # Merge and save
    all_datasets = [sciq_prompts, s2orc_prompts, arxiv_prompts, textbook_prompts]
    output_file = downloader.merge_and_save_datasets(all_datasets, "science_prompts.pkl")
    
    logger.info(f"\nDataset preparation complete! Output saved to: {output_file}")
    
    # Create a small test set
    with open(output_file, 'rb') as f:
        all_prompts = pickle.load(f)
    
    test_prompts = all_prompts[:100]  # First 100 for testing
    test_file = os.path.join(downloader.output_dir, "science_prompts_test.pkl")
    with open(test_file, 'wb') as f:
        pickle.dump(test_prompts, f)
    
    logger.info(f"Created test set with 100 prompts: {test_file}")


if __name__ == "__main__":
    main()
