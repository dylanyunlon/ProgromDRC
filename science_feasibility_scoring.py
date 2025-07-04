"""
Science Feasibility Scoring Module for CDRC Framework
Evaluates scientific validity and feasibility of generated text
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict, Tuple, Optional
import re
import json
import logging
from dataclasses import dataclass
import sympy as sp
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScienceEvaluationResult:
    """Container for science evaluation results"""
    overall_score: float
    physics_validity: float
    chemistry_validity: float
    logical_coherence: float
    terminology_accuracy: float
    formula_correctness: float
    experimental_feasibility: float
    confidence: float
    violations: List[str]
    field: str


class ScientificRuleChecker:
    """Rule-based checker for basic scientific law violations"""
    
    def __init__(self):
        self.physics_rules = {
            'conservation_energy': [
                r'energy.*created',
                r'energy.*destroyed',
                r'perpetual motion',
                r'free energy.*from nothing'
            ],
            'conservation_mass': [
                r'mass.*disappear',
                r'matter.*created.*nothing',
                r'atoms.*destroyed'
            ],
            'thermodynamics': [
                r'entropy.*decrease.*isolated',
                r'heat.*cold.*hot.*spontaneous',
                r'100%.*efficient'
            ],
            'relativity': [
                r'faster than.*light',
                r'exceed.*speed.*light',
                r'instant.*communication'
            ]
        }
        
        self.chemistry_rules = {
            'conservation': [
                r'atoms.*created.*reaction',
                r'elements.*transform.*another',
                r'lead.*gold.*chemical'
            ],
            'thermodynamics': [
                r'endothermic.*spontaneous.*room temp',
                r'exothermic.*absorb.*heat'
            ],
            'stoichiometry': [
                r'unbalanced.*equation.*correct'
            ]
        }
    
    def check_violations(self, text: str) -> List[str]:
        """Check text for scientific law violations"""
        violations = []
        text_lower = text.lower()
        
        # Check physics rules
        for rule_type, patterns in self.physics_rules.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    violations.append(f"Physics violation: {rule_type}")
                    break
        
        # Check chemistry rules
        for rule_type, patterns in self.chemistry_rules.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    violations.append(f"Chemistry violation: {rule_type}")
                    break
        
        return violations


class FormulaValidator:
    """Validates mathematical and chemical formulas"""
    
    def __init__(self):
        self.chemical_elements = set([
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            # ... add all elements
        ])
    
    def validate_chemical_formula(self, formula: str) -> Tuple[bool, str]:
        """Validate chemical formula"""
        # Remove spaces
        formula = formula.replace(' ', '')
        
        # Basic pattern for chemical formulas
        pattern = r'^([A-Z][a-z]?\d*)+$'
        
        if not re.match(pattern, formula):
            return False, "Invalid formula format"
        
        # Extract elements
        elements = re.findall(r'[A-Z][a-z]?', formula)
        
        # Check if all elements are valid
        for element in elements:
            if element not in self.chemical_elements:
                return False, f"Unknown element: {element}"
        
        return True, "Valid chemical formula"
    
    def validate_mathematical_expression(self, expr: str) -> Tuple[bool, str]:
        """Validate mathematical expression using SymPy"""
        try:
            # Parse expression
            parsed = sp.sympify(expr)
            return True, "Valid mathematical expression"
        except Exception as e:
            return False, f"Invalid expression: {str(e)}"


class ScienceFeasibilityScorer:
    def __init__(self, 
                 base_model="allenai/scibert_scivocab_uncased",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 device="cuda",
                 use_gpt4=False,
                 gpt4_api_key=None):
        """
        Initialize science feasibility scorer
        
        Args:
            base_model: Base model for science text understanding
            embedding_model: Model for semantic similarity
            device: Device to run models on
            use_gpt4: Whether to use GPT-4 for advanced evaluation
            gpt4_api_key: OpenAI API key if using GPT-4
        """
        self.device = device
        self.use_gpt4 = use_gpt4
        
        if use_gpt4 and gpt4_api_key:
            openai.api_key = gpt4_api_key
        
        # Load SciBERT for science text understanding
        logger.info(f"Loading SciBERT model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.scibert_model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3  # Valid, Invalid, Uncertain
        ).to(device)
        
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize specialized models
        self.rule_checker = ScientificRuleChecker()
        self.formula_validator = FormulaValidator()
        
        # Load domain-specific models if available
        self._load_domain_models()
        
        # Knowledge base embeddings (to be loaded)
        self.knowledge_embeddings = None
        self.knowledge_texts = None
    
    def _load_domain_models(self):
        """Load domain-specific models"""
        try:
            # Try loading Galactica for scientific reasoning
            self.galactica = AutoModelForCausalLM.from_pretrained(
                "facebook/galactica-1.3b",
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.galactica_tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
            logger.info("Loaded Galactica model for scientific reasoning")
        except:
            logger.warning("Could not load Galactica model")
            self.galactica = None
            
        try:
            # Try loading ChemBERTa for chemistry
            self.chemberta = AutoModelForSequenceClassification.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1",
                num_labels=2
            ).to(self.device)
            self.chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            logger.info("Loaded ChemBERTa for chemistry evaluation")
        except:
            logger.warning("Could not load ChemBERTa model")
            self.chemberta = None
    
    def load_knowledge_base(self, knowledge_file: str):
        """Load scientific knowledge base for similarity checking"""
        logger.info(f"Loading knowledge base from {knowledge_file}")
        
        with open(knowledge_file, 'r') as f:
            knowledge_data = json.load(f)
        
        self.knowledge_texts = knowledge_data['texts']
        self.knowledge_embeddings = self.embedding_model.encode(
            self.knowledge_texts,
            batch_size=32,
            show_progress_bar=True
        )
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge statements")
    
    def create_biased_evaluator(self, bias_level: float = 0.15) -> 'BiasedScienceScorer':
        """
        Create a biased version of the scorer
        
        Args:
            bias_level: Level of bias (0.15, 0.30, 0.70)
            
        Returns:
            Biased scorer instance
        """
        return BiasedScienceScorer(self, bias_level)
    
    def evaluate_scientific_validity(self, 
                                   prompt: str, 
                                   continuation: str,
                                   field: str = "general") -> ScienceEvaluationResult:
        """
        Comprehensive evaluation of scientific validity
        
        Args:
            prompt: Input prompt
            continuation: Generated continuation
            field: Scientific field (physics/chemistry/general)
            
        Returns:
            Detailed evaluation results
        """
        full_text = prompt + " " + continuation
        
        # 1. Rule-based violation checking
        violations = self.rule_checker.check_violations(full_text)
        
        # 2. Formula validation
        formula_score = self._evaluate_formulas(continuation)
        
        # 3. Logical coherence
        coherence_score = self._evaluate_coherence(prompt, continuation)
        
        # 4. Terminology accuracy
        terminology_score = self._evaluate_terminology(continuation, field)
        
        # 5. Domain-specific evaluation
        if field == "chemistry":
            domain_score = self._evaluate_chemistry(full_text)
        elif field == "physics":
            domain_score = self._evaluate_physics(full_text)
        else:
            domain_score = self._evaluate_general_science(full_text)
        
        # 6. Experimental feasibility
        experimental_score = self._evaluate_experimental_feasibility(continuation)
        
        # 7. GPT-4 evaluation (if enabled)
        if self.use_gpt4:
            gpt4_score = self._evaluate_with_gpt4(prompt, continuation, field)
        else:
            gpt4_score = domain_score
        
        # Combine scores
        if violations:
            # Severe penalty for law violations
            overall_score = 0.1
            confidence = 0.9
        else:
            scores = [
                domain_score * 0.3,
                coherence_score * 0.2,
                terminology_score * 0.15,
                formula_score * 0.15,
                experimental_score * 0.1,
                gpt4_score * 0.1
            ]
            overall_score = sum(scores)
            confidence = self._calculate_confidence(scores)
        
        return ScienceEvaluationResult(
            overall_score=overall_score,
            physics_validity=domain_score if field == "physics" else 0.5,
            chemistry_validity=domain_score if field == "chemistry" else 0.5,
            logical_coherence=coherence_score,
            terminology_accuracy=terminology_score,
            formula_correctness=formula_score,
            experimental_feasibility=experimental_score,
            confidence=confidence,
            violations=violations,
            field=field
        )
    
    def _evaluate_formulas(self, text: str) -> float:
        """Evaluate correctness of formulas in text"""
        # Extract potential formulas
        math_patterns = [
            r'\$([^\$]+)\$',  # LaTeX inline
            r'\\\[([^\]]+)\\\]',  # LaTeX display
            r'([A-Za-z0-9]+\s*=\s*[A-Za-z0-9\+\-\*/\^\s]+)',  # Simple equations
        ]
        
        chem_pattern = r'([A-Z][a-z]?\d*)+(?:\s*\+\s*([A-Z][a-z]?\d*)+)*\s*(?:->|→)\s*([A-Z][a-z]?\d*)+'
        
        formulas_found = []
        scores = []
        
        # Check mathematical formulas
        for pattern in math_patterns:
            formulas_found.extend(re.findall(pattern, text))
        
        for formula in formulas_found:
            valid, _ = self.formula_validator.validate_mathematical_expression(formula)
            scores.append(1.0 if valid else 0.0)
        
        # Check chemical formulas
        chem_formulas = re.findall(chem_pattern, text)
        for formula_parts in chem_formulas:
            # Basic validation
            scores.append(0.8)  # Simplified - would need proper balancing check
        
        if not scores:
            return 1.0  # No formulas found, no penalty
        
        return np.mean(scores)
    
    def _evaluate_coherence(self, prompt: str, continuation: str) -> float:
        """Evaluate logical coherence between prompt and continuation"""
        # Semantic similarity
        prompt_embedding = self.embedding_model.encode(prompt)
        continuation_embedding = self.embedding_model.encode(continuation)
        
        similarity = np.dot(prompt_embedding, continuation_embedding) / (
            np.linalg.norm(prompt_embedding) * np.linalg.norm(continuation_embedding)
        )
        
        # Check for logical connectives
        connectives = ['therefore', 'thus', 'because', 'since', 'as a result']
        has_connective = any(conn in continuation.lower() for conn in connectives)
        
        coherence_score = similarity * 0.7 + (0.3 if has_connective else 0.0)
        
        return min(coherence_score, 1.0)
    
    def _evaluate_terminology(self, text: str, field: str) -> float:
        """Evaluate scientific terminology usage"""
        # Use SciBERT to evaluate terminology
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.scibert_model(**inputs)
            # Simplified: use model confidence as terminology score
            probs = torch.softmax(outputs.logits, dim=-1)
            terminology_score = probs.max().item()
        
        return terminology_score
    
    def _evaluate_chemistry(self, text: str) -> float:
        """Chemistry-specific evaluation"""
        if self.chemberta is None:
            return 0.5  # Default score if model not available
        
        inputs = self.chemberta_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.chemberta(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            validity_score = probs[0, 1].item()  # Assuming index 1 is "valid"
        
        return validity_score
    
    def _evaluate_physics(self, text: str) -> float:
        """Physics-specific evaluation"""
        # Check for common physics concepts
        physics_concepts = [
            'energy', 'force', 'momentum', 'velocity', 'acceleration',
            'mass', 'charge', 'field', 'wave', 'particle'
        ]
        
        concept_count = sum(1 for concept in physics_concepts if concept in text.lower())
        concept_score = min(concept_count / 3, 1.0)  # Normalize
        
        # Use Galactica if available
        if self.galactica is not None:
            validity_score = self._evaluate_with_galactica(text)
        else:
            validity_score = concept_score
        
        return validity_score
    
    def _evaluate_general_science(self, text: str) -> float:
        """General scientific evaluation"""
        # Combine physics and chemistry evaluations
        physics_score = self._evaluate_physics(text)
        chemistry_score = self._evaluate_chemistry(text)
        
        # Weight based on detected field
        chem_keywords = sum(1 for kw in ['reaction', 'compound', 'molecule'] if kw in text.lower())
        phys_keywords = sum(1 for kw in ['force', 'energy', 'field'] if kw in text.lower())
        
        if chem_keywords > phys_keywords:
            return chemistry_score * 0.7 + physics_score * 0.3
        elif phys_keywords > chem_keywords:
            return physics_score * 0.7 + chemistry_score * 0.3
        else:
            return (physics_score + chemistry_score) / 2
    
    def _evaluate_experimental_feasibility(self, text: str) -> float:
        """Evaluate if described experiments are feasible"""
        # Keywords indicating experimental procedures
        experimental_keywords = [
            'experiment', 'measure', 'observe', 'test', 'procedure',
            'apparatus', 'equipment', 'method', 'protocol'
        ]
        
        if not any(kw in text.lower() for kw in experimental_keywords):
            return 1.0  # No experimental content, no penalty
        
        # Check for impossible conditions
        impossible_patterns = [
            r'temperature.*below.*absolute zero',
            r'pressure.*negative',
            r'measure.*simultaneously.*position.*momentum.*exactly',
            r'100%.*yield',
            r'perpetual.*motion'
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, text.lower()):
                return 0.0
        
        # Check for realistic conditions
        realistic_score = 0.8  # Default
        
        # Temperature checks
        temp_match = re.search(r'(\d+)\s*(K|°C|°F|kelvin|celsius)', text.lower())
        if temp_match:
            temp_value = float(temp_match.group(1))
            unit = temp_match.group(2)
            
            if unit in ['K', 'kelvin'] and (temp_value < 0 or temp_value > 10000):
                realistic_score *= 0.5
            elif unit in ['°C', 'celsius'] and (temp_value < -273 or temp_value > 5000):
                realistic_score *= 0.5
        
        return realistic_score
    
    def _evaluate_with_galactica(self, text: str) -> float:
        """Use Galactica model for evaluation"""
        prompt = f"Evaluate the scientific accuracy of the following statement:\n{text}\n\nAccuracy:"
        
        inputs = self.galactica_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.galactica.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.1,
                do_sample=False
            )
        
        response = self.galactica_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response for accuracy score
        if "high" in response.lower() or "correct" in response.lower():
            return 0.9
        elif "medium" in response.lower() or "mostly" in response.lower():
            return 0.6
        else:
            return 0.3
    
    def _evaluate_with_gpt4(self, prompt: str, continuation: str, field: str) -> float:
        """Use GPT-4 for advanced evaluation"""
        if not self.use_gpt4:
            return 0.5
        
        evaluation_prompt = f"""
        Evaluate the scientific accuracy and feasibility of the following text continuation.
        Field: {field}
        
        Prompt: {prompt}
        Continuation: {continuation}
        
        Evaluate on a scale of 0-1 considering:
        1. Scientific accuracy
        2. Logical coherence
        3. Feasibility of any described phenomena/experiments
        4. Correct use of terminology
        
        Respond with only a number between 0 and 1.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific accuracy evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response['choices'][0]['message']['content'].strip()
            score = float(score_text)
            return min(max(score, 0.0), 1.0)
        except:
            logger.warning("GPT-4 evaluation failed, using fallback")
            return 0.5
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence in the evaluation"""
        # Higher variance = lower confidence
        variance = np.var(scores)
        confidence = 1.0 - min(variance * 2, 0.5)
        
        # Adjust based on average score extremity
        avg_score = np.mean(scores)
        if avg_score < 0.2 or avg_score > 0.8:
            confidence *= 1.1  # More confident in extreme cases
        
        return min(confidence, 1.0)
    
    def score_candidate_sets(self, x_cal: Dict, y_cal: Dict, 
                           prompts_data: Optional[List[Dict]] = None) -> Dict:
        """
        Score all candidate sets for scientific validity
        
        Args:
            x_cal: Calibration features dictionary
            y_cal: Calibration candidate sets
            prompts_data: Original prompts data with field information
            
        Returns:
            Updated x_cal with science scores
        """
        # Create mapping of idx to field if prompts_data provided
        idx_to_field = {}
        if prompts_data:
            for prompt in prompts_data:
                idx_to_field[prompt['idx']] = prompt.get('field', 'general')
        
        for idx, data in tqdm(x_cal.items(), desc="Scoring scientific validity"):
            texts = data['pred']
            
            # Get field for this prompt
            field = idx_to_field.get(idx, 'general')
            
            # Get original prompt (assuming it's stored somewhere)
            # For now, we'll use a placeholder
            prompt = "Scientific prompt"  # This should come from actual data
            
            # Score each continuation
            scores = []
            detailed_results = []
            
            for text in texts:
                result = self.evaluate_scientific_validity(prompt, text, field)
                scores.append(result.overall_score)
                detailed_results.append(result)
            
            # Store scores in x_cal
            x_cal[idx]['science_validity_scores'] = scores
            x_cal[idx]['science_detailed_results'] = detailed_results
            x_cal[idx]['field'] = field
        
        return x_cal


class BiasedScienceScorer:
    """Biased version of science scorer for testing robustness"""
    
    def __init__(self, base_scorer: ScienceFeasibilityScorer, bias_level: float):
        self.base_scorer = base_scorer
        self.bias_level = bias_level
        
    def score(self, prompt: str, continuation: str, field: str = "general") -> float:
        """Score with intentional bias"""
        base_result = self.base_scorer.evaluate_scientific_validity(prompt, continuation, field)
        base_score = base_result.overall_score
        
        # Apply bias: randomly flip some scores
        if np.random.random() < self.bias_level:
            # Invert score
            biased_score = 1.0 - base_score
        else:
            biased_score = base_score
        
        # Add noise
        noise = np.random.normal(0, 0.1 * self.bias_level)
        biased_score = np.clip(biased_score + noise, 0.0, 1.0)
        
        return biased_score


# Example usage
if __name__ == "__main__":
    # Initialize scorer
    scorer = ScienceFeasibilityScorer(use_gpt4=False)
    
    # Test examples
    test_cases = [
        {
            "prompt": "When temperature increases,",
            "continuation": "the reaction rate typically increases due to higher molecular kinetic energy.",
            "field": "chemistry"
        },
        {
            "prompt": "According to conservation of energy,",
            "continuation": "energy can be created from nothing in special circumstances.",
            "field": "physics"
        },
        {
            "prompt": "In an exothermic reaction,",
            "continuation": "heat is released to the surroundings, making the container feel warm.",
            "field": "chemistry"
        }
    ]
    
    for test in test_cases:
        result = scorer.evaluate_scientific_validity(
            test["prompt"], 
            test["continuation"], 
            test["field"]
        )
        
        print(f"\nPrompt: {test['prompt']}")
        print(f"Continuation: {test['continuation']}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Violations: {result.violations}")
        print(f"Confidence: {result.confidence:.3f}")
