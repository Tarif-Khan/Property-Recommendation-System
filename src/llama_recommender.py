import numpy as np
import pandas as pd
import json
import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import re
from tqdm import tqdm
import time

# Hugging Face & Unsloth imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from unsloth import FastLanguageModel
from peft import LoraConfig
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaPropertyRecommender:
    """
    LLAMA-based property recommendation system using Unsloth for fine-tuning.
    """
    
    def __init__(self, 
                 model_name: str = "unsloth/llama-3-8b-bnb-4bit",
                 cache_dir: Optional[str] = None,
                 max_length: int = 1024,
                 device: str = "auto"):
        """
        Initialize the LLAMA-based property recommender.
        
        Args:
            model_name: Name of the LLAMA model to use
            cache_dir: Directory to cache model and embeddings
            max_length: Maximum token length for input sequences
            device: Device to use for inference ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = max_length
        self.device = self._get_device() if device == "auto" else device
        
        self.tokenizer = None
        self.model = None
        self.is_fit = False
        self.property_data = None
        self.location_data = {}  # To store location information for each property
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _initialize_model(self):
        """Initialize the LLAMA model and tokenizer."""
        logger.info(f"Initializing LLAMA model: {self.model_name}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Configure QLoRA for efficient fine-tuning
        # Using Unsloth's optimized loading
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            cache_dir=self.cache_dir
        )
        
        # Update tokenizer if Unsloth returns a new one
        if tokenizer is not None:
            self.tokenizer = tokenizer
            
        # Save the model
        self.model = model
        
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model initialized successfully")
    
    def _property_to_text(self, prop: Dict[str, Any]) -> str:
        """
        Convert a property dictionary to a text description.
        
        Args:
            prop: Property dictionary
            
        Returns:
            Text description of the property
        """
        # Extract subject information
        subject = prop.get('subject', {})
        
        # Build a detailed property description
        description = "Property Description:\n"
        
        # Address and location
        address = subject.get('address', 'unknown address')
        description += f"Address: {address}\n"
        
        municipality = subject.get('municipality_district', '')
        if municipality:
            description += f"Municipality/District: {municipality}\n"
            
        # Property details
        structure_type = subject.get('structure_type', '')
        if structure_type:
            description += f"Structure Type: {structure_type}\n"
            
        style = subject.get('style', '')
        if style:
            description += f"Style: {style}\n"
            
        year_built = subject.get('year_built', '')
        if year_built:
            description += f"Year Built: {year_built}\n"
            
        effective_age = subject.get('effective_age', '')
        if effective_age:
            description += f"Effective Age: {effective_age}\n"
            
        condition = subject.get('condition', '')
        if condition:
            description += f"Condition: {condition}\n"
        
        # Size information
        lot_size = subject.get('lot_size_sf', '')
        if lot_size and lot_size != 'n/a':
            description += f"Lot Size: {lot_size} sq ft\n"
            
        gla = subject.get('gla', '')
        if gla:
            description += f"Gross Living Area: {gla} sq ft\n"
            
        main_lvl_area = subject.get('main_lvl_area', '')
        if main_lvl_area:
            description += f"Main Level Area: {main_lvl_area} sq ft\n"
            
        second_lvl_area = subject.get('second_lvl_area', '')
        if second_lvl_area:
            description += f"Second Level Area: {second_lvl_area} sq ft\n"
            
        third_lvl_area = subject.get('third_lvl_area', '')
        if third_lvl_area:
            description += f"Third Level Area: {third_lvl_area} sq ft\n"
            
        basement_area = subject.get('basement_area', '')
        if basement_area:
            description += f"Basement Area: {basement_area} sq ft\n"
        
        # Room information
        rooms = subject.get('room_count', '')
        if rooms:
            description += f"Room Count: {rooms}\n"
            
        beds = subject.get('num_beds', '')
        if beds:
            description += f"Bedrooms: {beds}\n"
            
        baths = subject.get('num_baths', '')
        if baths:
            description += f"Bathrooms: {baths}\n"
        
        # Building features
        construction = subject.get('construction', '')
        if construction:
            description += f"Construction: {construction}\n"
            
        basement = subject.get('basement', '')
        if basement:
            description += f"Basement: {basement}\n"
            
        heating = subject.get('heating', '')
        if heating:
            description += f"Heating: {heating}\n"
            
        cooling = subject.get('cooling', '')
        if cooling:
            description += f"Cooling: {cooling}\n"
            
        return description.strip()
    
    def _extract_location_data(self, properties: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Extract location data from properties for filtering.
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            Dictionary mapping order IDs to location information
        """
        location_data = {}
        
        for prop in properties:
            order_id = prop.get('orderID')
            subject = prop.get('subject', {})
            address = subject.get('address', '')
            
            # Extract location information
            city = ''
            province = ''
            postal_code = ''
            municipality = subject.get('municipality_district', '')
            
            # Extract postal code (Canadian format: A1A 1A1)
            postal_pattern = r'[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d'
            postal_match = re.search(postal_pattern, address)
            if postal_match:
                postal_code = postal_match.group(0)
            
            # Extract province (assuming standard Canadian province abbreviations)
            provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
            for prov in provinces:
                if f" {prov} " in f" {address} ":
                    province = prov
                    break
            
            # Extract city (simple approach)
            if province and province in address:
                parts = address.split(province)[0].strip().split()
                if parts:
                    city = parts[-1]
            
            location_data[order_id] = {
                'city': city,
                'province': province,
                'postal_code': postal_code,
                'municipality': municipality
            }
            
        return location_data
    
    def _create_training_examples(self, properties: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Create training examples for the LLAMA model.
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            List of training examples
        """
        training_data = []
        
        # Extract properties that have comparable properties (comps)
        for prop in properties:
            comps = prop.get('comps', [])
            if not comps:
                continue
            
            subject_text = self._property_to_text(prop)
            
            # Create prompt-response pairs for each comparable property
            for comp in comps:
                # Convert comp to property format for _property_to_text
                comp_prop = {
                    'subject': {
                        'address': comp.get('address', '') + ' ' + comp.get('city_province', ''),
                        'structure_type': comp.get('prop_type', ''),
                        'style': comp.get('stories', ''),
                        # Other fields may be missing for comps, so we use empty strings
                        'year_built': comp.get('year_built', ''),
                        'gla': comp.get('gla', ''),
                        'lot_size_sf': comp.get('lot_size', ''),
                        'condition': comp.get('condition', ''),
                        'room_count': comp.get('rooms', ''),
                        'num_beds': comp.get('bedrooms', ''),
                        'num_baths': comp.get('bathrooms', ''),
                    }
                }
                
                # Create a formatted instruction for the model
                instruction = (
                    f"Given the following property details, identify a comparable property "
                    f"that would be a good match for valuation purposes:\n\n{subject_text}"
                )
                
                # The response should include details about why this comp is suitable
                response = (
                    f"A comparable property for valuation purposes is:\n\n"
                    f"Address: {comp.get('address', '')} {comp.get('city_province', '')}\n"
                    f"Property Type: {comp.get('prop_type', '')}\n"
                    f"Style: {comp.get('stories', '')}\n"
                    f"Sale Date: {comp.get('sale_date', '')}\n"
                    f"Sale Price: {comp.get('sale_price', '')}\n"
                    f"Distance from Subject: {comp.get('distance_to_subject', '')}\n\n"
                    f"This property is a good comparable because it is of similar type and style, "
                    f"located within {comp.get('distance_to_subject', '')} of the subject property."
                )
                
                # Add to training data
                training_data.append({
                    "instruction": instruction,
                    "output": response
                })
        
        return training_data
    
    def _prepare_dataset(self, training_examples: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            training_examples: List of training examples
            
        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Preparing dataset with {len(training_examples)} examples")
        
        # Convert to Dataset format
        dataset = Dataset.from_list(training_examples)
        
        # Format dataset for instruction fine-tuning
        def format_instruction(example):
            formatted_text = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
            return {"text": formatted_text}
        
        dataset = dataset.map(format_instruction)
        return dataset
    
    def fit(self, properties: List[Dict[str, Any]], 
            lr: float = 2e-4, 
            num_epochs: int = 3, 
            batch_size: int = 4,
            lora_rank: int = 16) -> None:
        """
        Fine-tune the LLAMA model on property data.
        
        Args:
            properties: List of property dictionaries
            lr: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for training
            lora_rank: Rank for LoRA adapters
        """
        # Store property data
        self.property_data = properties
        
        # Extract location data
        self.location_data = self._extract_location_data(properties)
        
        # Initialize the model if not already initialized
        if self.model is None:
            self._initialize_model()
        
        # Create training examples
        logger.info("Creating training examples from property data")
        training_examples = self._create_training_examples(properties)
        
        if not training_examples:
            logger.warning("No training examples could be created. Unable to fine-tune the model.")
            return
            
        logger.info(f"Created {len(training_examples)} training examples")
        
        # Prepare dataset
        dataset = self._prepare_dataset(training_examples)
        
        # Configure LoRA for fine-tuning
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for fine-tuning
        logger.info("Preparing model for fine-tuning")
        ft_model = FastLanguageModel.get_peft_model(
            self.model,
            lora_config,
            train_on_inputs=False
        )
        
        # Start fine-tuning
        logger.info(f"Starting fine-tuning for {num_epochs} epochs")
        try:
            trainer = FastLanguageModel.get_trainer(
                model=ft_model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.max_length,
                batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                logging_steps=50
            )
            
            # Train the model
            trainer.train()
            
            # Save the fine-tuned model if cache_dir is specified
            if self.cache_dir:
                save_path = self.cache_dir / "fine_tuned_llama"
                trainer.save_model(save_path)
                logger.info(f"Fine-tuned model saved to {save_path}")
            
            # Set is_fit to True
            self.is_fit = True
            logger.info("Fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def _matches_location_criteria(self, 
                                  order_id: str, 
                                  location_params: Dict[str, str]) -> bool:
        """
        Check if a property matches location criteria.
        
        Args:
            order_id: Property order ID
            location_params: Location parameters for filtering
            
        Returns:
            True if property matches criteria, False otherwise
        """
        if not location_params:
            return True
            
        prop_location = self.location_data.get(order_id, {})
        
        # Check city
        if location_params.get('city') and location_params['city'].lower() != prop_location.get('city', '').lower():
            return False
            
        # Check province
        if location_params.get('province') and location_params['province'].upper() != prop_location.get('province', '').upper():
            return False
            
        # Check postal code (partial match)
        if location_params.get('postal_code') and not prop_location.get('postal_code', '').startswith(location_params['postal_code']):
            return False
            
        # Check municipality
        if location_params.get('municipality') and location_params['municipality'].lower() not in prop_location.get('municipality', '').lower():
            return False
            
        return True
    
    def recommend(self, 
                 query_property: Dict[str, Any], 
                 top_n: int = 5,
                 filter_by_location: bool = True,
                 location_params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend similar properties using the fine-tuned LLAMA model.
        
        Args:
            query_property: Query property dictionary
            top_n: Number of recommendations to return
            filter_by_location: Whether to filter by location
            location_params: Location parameters for filtering
            
        Returns:
            List of recommended properties
        """
        if not self.is_fit:
            logger.error("Model has not been fine-tuned yet. Please call fit() first.")
            return []
            
        # Apply location filtering if requested
        filtered_properties = []
        if filter_by_location and location_params:
            for prop in self.property_data:
                if self._matches_location_criteria(prop.get('orderID', ''), location_params):
                    filtered_properties.append(prop)
        else:
            filtered_properties = self.property_data
            
        if not filtered_properties:
            logger.warning("No properties match the location criteria.")
            return []
            
        # Prepare the query
        query_text = self._property_to_text(query_property)
        prompt = f"<s>[INST] Given the following property details, identify {top_n} comparable properties that would be good matches for valuation purposes:\n\n{query_text} [/INST]"
        
        # Generate recommendations using the fine-tuned model
        logger.info("Generating recommendations with fine-tuned LLAMA model")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Parse the response to extract recommendations
        # This is a simple implementation and might need refinement based on actual model outputs
        recommendations = []
        prop_sections = re.split(r'\n\s*\n', response)
        
        for section in prop_sections:
            if not section.strip():
                continue
                
            # Try to extract address and other details
            address_match = re.search(r'Address:\s*(.*?)(?:\n|$)', section)
            prop_type_match = re.search(r'Property Type:\s*(.*?)(?:\n|$)', section)
            
            if address_match:
                address = address_match.group(1).strip()
                
                # Find matching property in our dataset
                matching_props = [p for p in filtered_properties if address.lower() in p.get('subject', {}).get('address', '').lower()]
                
                if matching_props:
                    recommendations.append(matching_props[0])
                else:
                    # If no exact match, try to find a property with similar characteristics
                    prop_type = prop_type_match.group(1).strip() if prop_type_match else ""
                    
                    similar_props = [p for p in filtered_properties 
                                    if prop_type.lower() in p.get('subject', {}).get('structure_type', '').lower()]
                    
                    if similar_props and len(recommendations) < top_n:
                        recommendations.append(similar_props[0])
            
            # Break if we have enough recommendations
            if len(recommendations) >= top_n:
                break
                
        # If we couldn't extract enough recommendations from the model output,
        # fall back to simple similarity-based recommendations
        if len(recommendations) < top_n:
            logger.info("Adding fallback recommendations based on property type similarity")
            
            query_type = query_property.get('subject', {}).get('structure_type', '')
            remaining_props = [p for p in filtered_properties if p not in recommendations]
            
            for prop in remaining_props:
                if len(recommendations) >= top_n:
                    break
                    
                prop_type = prop.get('subject', {}).get('structure_type', '')
                if prop_type == query_type and prop != query_property:
                    recommendations.append(prop)
        
        return recommendations[:top_n]
    
    def generate_explanation(self, 
                           query_property: Dict[str, Any], 
                           recommendation: Dict[str, Any],
                           detailed: bool = False) -> str:
        """
        Generate an explanation for why a property was recommended.
        
        Args:
            query_property: Query property dictionary
            recommendation: Recommended property dictionary
            detailed: Whether to generate a detailed explanation
            
        Returns:
            Explanation string
        """
        if not self.is_fit:
            return "Model has not been fine-tuned yet."
            
        query_text = self._property_to_text(query_property)
        rec_text = self._property_to_text(recommendation)
        
        # Create a prompt for explanation generation
        prompt = f"<s>[INST] I need to explain why the following property:\n\n{rec_text}\n\nis a good comparable for this property:\n\n{query_text}\n\nPlease provide a {'detailed' if detailed else 'concise'} explanation of why these properties are comparable for valuation purposes. [/INST]"
        
        # Generate explanation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512 if detailed else 256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
            
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation.replace(prompt, "").strip()
        
        return explanation 