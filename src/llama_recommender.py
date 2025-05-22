import logging
import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the required modules for LLAMA model
try:
    from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig # get_peft_model is part of FastLanguageModel with unsloth
    from unsloth import FastLanguageModel # Unsloth's main import
except ImportError:
    logger.error(
        "Failed to import necessary libraries (transformers, peft, unsloth). "
        "Please ensure they are installed. Functionality will be severely limited."
    )
    # Define dummy classes or raise an error to prevent further execution if critical imports fail
    class FastLanguageModel: pass # Dummy
    class LoraConfig: pass # Dummy
    # Depending on the desired behavior, you might want to raise an exception here.

class LlamaPropertyRecommender:
    """
    LLAMA-based property recommendation system using Unsloth for fine-tuning.
    """
    
    def __init__(self, 
                 model_name: str = "unsloth/llama-2-7b-hf-unsloth", # Example Unsloth compatible model
                 cache_dir: Optional[str] = None,
                 max_length: int = 2048, # Increased max_length for potentially longer sequences
                 device: str = "auto"):
        """
        Initialize the LLAMA-based property recommender.
        
        Args:
            model_name: Name of the Unsloth-compatible LLAMA model to use.
            cache_dir: Directory to cache model, fine-tuned adapters, and tokenizer.
            max_length: Maximum token length for input sequences.
            device: Device to use for inference ('cuda', 'cpu', or 'auto').
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = max_length
        self.device = self._get_device() if device == "auto" else device
        
        self.tokenizer = None
        self.model = None # This will be the PEFT model after fine-tuning
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
        """Initialize the LLAMA model and tokenizer using Unsloth."""
        logger.info(f"Initializing LLAMA model: {self.model_name} using Unsloth")
        
        try:
            # Load model and tokenizer using Unsloth's FastLanguageModel
            # This handles 4-bit quantization (QLoRA) automatically if load_in_4bit=True
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_length,
                dtype=None,  # Unsloth handles dtype automatically for QLoRA
                load_in_4bit=True, # Enable QLoRA
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                # token= "hf_...", # Add Hugging Face token if needed for private models
            )
            self.model = model # Store the base model for now, will be wrapped by PEFT later
            self.tokenizer = tokenizer

            # Set padding token if not already set (LLAMA models usually use EOS as PAD)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            logger.info("Unsloth FastLanguageModel and Tokenizer initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Unsloth model: {str(e)}")
            # Potentially re-raise or handle gracefully
            raise

    def _property_to_text(self, prop: Dict[str, Any]) -> str:
        """
        Convert a property dictionary to a text description.
        (This method seems well-defined, keeping it as is for now)
        """
        subject = prop.get('subject', {})
        description_parts = []
        
        def add_if_present(key, prefix, suffix=""):
            value = subject.get(key)
            if value and str(value).lower() != 'n/a':
                description_parts.append(f"{prefix}{value}{suffix}")

        description_parts.append(f"Address: {subject.get('address', 'unknown address')}")
        add_if_present('municipality_district', "Municipality/District: ")
        add_if_present('structure_type', "Structure Type: ")
        add_if_present('style', "Style: ")
        add_if_present('year_built', "Year Built: ")
        add_if_present('effective_age', "Effective Age: ")
        add_if_present('condition', "Condition: ")
        add_if_present('lot_size_sf', "Lot Size: ", " sq ft")
        add_if_present('gla', "Gross Living Area: ", " sq ft")
        add_if_present('main_lvl_area', "Main Level Area: ", " sq ft")
        add_if_present('second_lvl_area', "Second Level Area: ", " sq ft")
        add_if_present('third_lvl_area', "Third Level Area: ", " sq ft")
        add_if_present('basement_area', "Basement Area: ", " sq ft")
        add_if_present('room_count', "Room Count: ")
        add_if_present('num_beds', "Bedrooms: ")
        add_if_present('num_baths', "Bathrooms: ")
        add_if_present('construction', "Construction: ")
        add_if_present('basement', "Basement: ")
        add_if_present('heating', "Heating: ")
        add_if_present('cooling', "Cooling: ")
        
        return "Property Description:\n" + "\n".join(description_parts)

    def _extract_location_data(self, properties: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Extract location data. (Keeping as is)
        """
        location_data = {}
        for prop in properties:
            order_id = prop.get('orderID')
            if not order_id: continue
            subject = prop.get('subject', {})
            address = subject.get('address', '')
            
            city, province, postal_code = '', '', ''
            municipality = subject.get('municipality_district', '')
            
            postal_pattern = r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d' # Optional space
            postal_match = re.search(postal_pattern, address)
            if postal_match:
                postal_code = postal_match.group(0).replace(" ", "") # Normalize

            canadian_provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
            for prov_abbr in canadian_provinces:
                if re.search(r'\b' + prov_abbr + r'\b', address): # Word boundary
                    province = prov_abbr
                    break
            
            # Improved city extraction
            if address:
                parts = address.split(',')
                if len(parts) > 1:
                    # Often city is before province or postal code in a comma-separated list
                    potential_city_part = parts[-2 if province or postal_code else -1].strip()
                    # Avoid picking up street numbers or unit numbers as city
                    if not any(char.isdigit() for char in potential_city_part.split()[-1]):
                         city = potential_city_part.split()[-1]


            location_data[order_id] = {
                'city': city, 'province': province, 'postal_code': postal_code, 'municipality': municipality
            }
        return location_data

    def _create_training_examples(self, properties: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Create training examples for the LLAMA model.
        """
        training_data = []
        for prop in properties:
            comps = prop.get('comps', [])
            if not comps:
                continue
            
            subject_text = self._property_to_text(prop)
            
            for comp in comps:
                # Constructing comp_prop in a more robust way
                comp_prop_subject = {
                    'address': f"{comp.get('address', '')} {comp.get('city_province', '')}".strip(),
                    'structure_type': comp.get('prop_type', ''),
                    'style': comp.get('stories', ''),
                    'year_built': comp.get('year_built', ''),
                    'gla': comp.get('gla', ''),
                    'lot_size_sf': comp.get('lot_size', ''),
                    'condition': comp.get('condition', ''),
                    'room_count': comp.get('rooms', ''),
                    'num_beds': comp.get('bedrooms', ''),
                    'num_baths': comp.get('bathrooms', ''),
                }
                comp_text = self._property_to_text({'subject': comp_prop_subject})

                instruction = (
                    f"Given the primary property details:\n{subject_text}\n\n"
                    f"Identify and describe a comparable property (comp) that would be suitable for valuation purposes."
                )
                
                response = (
                    f"A good comparable property is:\n{comp_text}\n\n"
                    f"Key comparable features include its sale date ({comp.get('sale_date', 'N/A')}), "
                    f"sale price ({comp.get('sale_price', 'N/A')}), "
                    f"and proximity ({comp.get('distance_to_subject', 'N/A')}). "
                    f"This property is suitable because it shares similarities in type, style, and location, "
                    f"making it a relevant benchmark for valuation."
                )
                
                training_data.append({
                    "instruction": instruction.strip(),
                    "output": response.strip()
                })
        return training_data

    def _prepare_dataset(self, training_examples: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        Formats examples into Alpaca instruction format.
        """
        logger.info(f"Preparing dataset with {len(training_examples)} examples")
        
        # Using Unsloth's recommended formatting
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs       = examples["input"] # This field needs to be present or handled
            outputs      = examples["output"]
            texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                # Must add EOS_TOKEN, otherwise your model will overshoot
                text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }

        # Create a dummy 'input' field if not present in training_examples
        # For this task, the 'instruction' contains the subject property, so 'input' can be empty.
        processed_examples = []
        for ex in training_examples:
            processed_examples.append({
                "instruction": ex["instruction"],
                "input": "", # Or some other placeholder if needed
                "output": ex["output"]
            })

        dataset = Dataset.from_list(processed_examples)
        dataset = dataset.map(formatting_prompts_func, batched = True,)
        return dataset

    def fit(self, properties: List[Dict[str, Any]], 
            lr: float = 2e-4, 
            num_epochs: int = 1, # Reduced default epochs for faster initial testing
            batch_size: int = 2, # Adjusted batch size
            lora_rank: int = 16,
            lora_alpha: Optional[int] = None,
            lora_dropout: float = 0.05
            ) -> None:
        """
        Fine-tune the LLAMA model on property data using Unsloth and PEFT.
        """
        if not properties:
            logger.error("Cannot fit model: No properties provided.")
            return

        self.property_data = properties
        self.location_data = self._extract_location_data(properties)
        
        if self.model is None or self.tokenizer is None:
            self._initialize_model()
            if self.model is None or self.tokenizer is None: # Check again after initialization
                logger.error("Model or tokenizer failed to initialize. Cannot proceed with fitting.")
                return
        
        # Prepare the model for QLoRA fine-tuning using Unsloth's PEFT integration
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = lora_rank,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"], # Standard for Llama
            lora_alpha = lora_alpha if lora_alpha is not None else lora_rank * 2,
            lora_dropout = lora_dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for Unsloth version
            random_state = 3407,
            use_rslora = False,  # Rank Stable LoRA
            loftq_config = None, # LoftQ
        )
        logger.info("PEFT model configured for LoRA.")

        training_examples = self._create_training_examples(properties)
        if not training_examples:
            logger.warning("No training examples could be created. Unable to fine-tune the model.")
            self.is_fit = False # Ensure is_fit is False
            return
        logger.info(f"Created {len(training_examples)} training examples")
        
        dataset = self._prepare_dataset(training_examples)
        
        # TrainingArguments
        output_dir_str = str(self.cache_dir / "llama_training_checkpoints") if self.cache_dir else "./llama_training_checkpoints"
        
        training_args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 4, # Accumulate gradients
            warmup_steps = 10, # Unsloth default warmup
            num_train_epochs = num_epochs,
            learning_rate = lr,
            fp16 = not torch.cuda.is_bf16_supported(), # Use fp16 if bf16 not supported
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1, # Log frequently
            optim = "adamw_8bit", # Use 8-bit AdamW optimizer for memory efficiency
            weight_decay = 0.01, # Unsloth default
            lr_scheduler_type = "linear", # Unsloth default
            seed = 3407, # Unsloth default seed
            output_dir = output_dir_str,
            report_to = "tensorboard", # Or "wandb", "none"
            # gradient_checkpointing = True, # Already handled by get_peft_model with unsloth
            # ddp_find_unused_parameters = False, # If using DDP
        )
        
        # Data Collator for Language Modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Trainer
        trainer = Trainer(
            model = self.model, # This is the PEFT model
            tokenizer = self.tokenizer,
            args = training_args,
            train_dataset = dataset,
            data_collator = data_collator,
        )
        
        logger.info(f"Starting fine-tuning for {num_epochs} epochs with LoRA rank {lora_rank}")
        try:
            trainer.train()
            
            self.is_fit = True
            logger.info("Fine-tuning completed successfully.")

            # Save the fine-tuned LoRA adapters and tokenizer
            if self.cache_dir:
                final_save_path = self.cache_dir / "fine_tuned_llama_adapters"
                self.model.save_pretrained(str(final_save_path)) # Saves LoRA adapters
                self.tokenizer.save_pretrained(str(final_save_path))
                logger.info(f"Fine-tuned LoRA adapters and tokenizer saved to {final_save_path}")
            else:
                logger.info("Cache directory not specified. Fine-tuned model adapters not saved persistently.")

        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            self.is_fit = False
            raise
    
    def _matches_location_criteria(self, order_id: str, location_params: Dict[str, str]) -> bool:
        """Check if a property matches location criteria."""
        if not location_params or not order_id: return True
        prop_location = self.location_data.get(order_id, {})
        if not prop_location: return True # Default to true if no location data for property

        for key, val_param in location_params.items():
            if not val_param: continue # Skip empty filter criteria
            val_prop = prop_location.get(key, '').lower()
            val_param_lower = val_param.lower()

            if key == 'postal_code':
                if not val_prop.startswith(val_param_lower.replace(" ","")): return False
            elif val_param_lower != val_prop:
                return False
        return True

    def recommend(self, 
                 query_property: Dict[str, Any], 
                 top_n: int = 3, # Reduced top_n for more focused output
                 filter_by_location: bool = True,
                 location_params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend similar properties using the fine-tuned LLAMA model.
        """
        if not self.is_fit or self.model is None or self.tokenizer is None:
            logger.error("Model has not been fine-tuned or initialized properly. Please call fit() first.")
            # Optionally, try to load a pre-fine-tuned model if available
            if self.cache_dir and (self.cache_dir / "fine_tuned_llama_adapters").exists():
                try:
                    logger.info(f"Attempting to load pre-fine-tuned adapters from {self.cache_dir / 'fine_tuned_llama_adapters'}")
                    self._initialize_model() # Ensure base model is loaded
                    self.model = FastLanguageModel.from_pretrained(
                        model = self.model, # Pass the base model
                        model_name = str(self.cache_dir / "fine_tuned_llama_adapters"),
                        # Unsloth specific loading for adapters might be different, check docs
                        # For now, assuming it loads adapters if model_name is a path to adapters
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(str(self.cache_dir / "fine_tuned_llama_adapters"))
                    self.is_fit = True # Assume fit if loaded successfully
                    logger.info("Successfully loaded pre-fine-tuned adapters.")
                except Exception as e:
                    logger.error(f"Failed to load pre-fine-tuned adapters: {e}. Please fit the model.")
                    return []
            else:
                return []
            
        query_text = self._property_to_text(query_property)
        
        # Using the same Alpaca prompt structure, but only providing instruction and input
        # The model should generate the response part.
        alpaca_prompt_inference = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
        instruction = (
            f"Given the primary property details, identify up to {top_n} highly comparable properties (comps) "
            f"that would be excellent matches for valuation purposes. For each comp, provide its description and key comparable features."
        )
        # The input for inference is the query property text
        prompt = alpaca_prompt_inference.format(instruction, query_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length - 256).to(self.device) # Reserve space for generation
        
        logger.info("Generating recommendations with fine-tuned LLAMA model...")
        recommendations_data = []
        try:
            with torch.inference_mode():
                # Unsloth specific generation if available, otherwise standard generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512, # Max tokens for the generated comparable properties
                    temperature=0.6, # Slightly lower for more factual output
                    top_p=0.9,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id # Important for generation
                )
            
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # Remove the prompt from the generated text
            response_part = generated_text.split("### Response:")[-1].strip()
            logger.info(f"Raw model response:\n{response_part}")

            # --- Advanced Parsing Logic Needed Here ---
            # The current parsing is very basic and likely insufficient.
            # It needs to reliably extract multiple property descriptions from the free-form text.
            # This might involve:
            # 1. Defining a clearer output format during fine-tuning (e.g., JSON strings, specific delimiters).
            # 2. Using more sophisticated regex or a small parsing model if the output is complex.
            # 3. Iteratively prompting for one comp at a time if single-comp generation is more reliable.

            # Placeholder for parsing - this needs significant improvement
            # Attempt to split into property sections if model outputs them separated by "A good comparable property is:"
            comp_indicators = ["A good comparable property is:", "Another comparable property:", "Comparable property:"]
            
            current_response_text = response_part
            extracted_comps_texts = []

            for _ in range(top_n):
                best_match_idx = -1
                best_indicator = None
                for indicator in comp_indicators:
                    try:
                        idx = current_response_text.lower().index(indicator.lower())
                        if best_match_idx == -1 or idx < best_match_idx:
                            best_match_idx = idx
                            best_indicator = indicator
                    except ValueError:
                        continue
                
                if best_indicator and best_match_idx != -1:
                    start_text = best_match_idx + len(best_indicator)
                    # Try to find the end of this comp's description
                    next_comp_idx = -1
                    for ind_inner in comp_indicators:
                        try:
                            idx_inner = current_response_text.lower().index(ind_inner.lower(), start_text)
                            if next_comp_idx == -1 or idx_inner < next_comp_idx:
                                next_comp_idx = idx_inner
                        except ValueError:
                            continue
                    
                    comp_text_segment = current_response_text[start_text : next_comp_idx if next_comp_idx !=-1 else len(current_response_text)].strip()
                    extracted_comps_texts.append(comp_text_segment)
                    current_response_text = current_response_text[ (next_comp_idx if next_comp_idx != -1 else len(current_response_text)) : ]
                    if not current_response_text.strip(): break
                else:
                    # If no more indicators, assume the rest is the last comp or irrelevant
                    if current_response_text.strip() and not extracted_comps_texts: # If it's the only part
                         extracted_comps_texts.append(current_response_text)
                    break
            
            logger.info(f"Extracted {len(extracted_comps_texts)} potential comp texts from model output.")

            # Match extracted texts to actual properties in the dataset
            # This is a simplified matching based on address and then property type
            # It requires the model to output parsable addresses or unique identifiers.
            
            available_properties = self.property_data
            if filter_by_location and location_params:
                 available_properties = [p for p in self.property_data if self._matches_location_criteria(p.get('orderID'), location_params)]
            
            # Prevent recommending the query property itself
            query_order_id = query_property.get('orderID')
            available_properties = [p for p in available_properties if p.get('orderID') != query_order_id]


            for comp_text_segment in extracted_comps_texts:
                if len(recommendations_data) >= top_n: break
                
                # Try to find an address in the segment
                address_match = re.search(r"Address:\s*(.*?)(?:\n|$)", comp_text_segment, re.IGNORECASE)
                identified_address = address_match.group(1).strip() if address_match else None
                
                matched_prop = None
                if identified_address:
                    # Try to find by address first (more specific)
                    for prop_candidate in available_properties:
                        candidate_addr = prop_candidate.get('subject', {}).get('address', '').lower()
                        if identified_address.lower() in candidate_addr:
                            # Avoid re-adding already found properties
                            if not any(rec.get('orderID') == prop_candidate.get('orderID') for rec in recommendations_data):
                                matched_prop = prop_candidate
                                break
                
                if not matched_prop:
                    # Fallback: Try to match based on other details if address matching fails or is absent.
                    # This is very crude and needs the model to output consistent details.
                    # For example, matching by property type mentioned in the text.
                    type_match = re.search(r"(?:Structure Type|Property Type):\s*(.*?)(?:\n|$)", comp_text_segment, re.IGNORECASE)
                    identified_type = type_match.group(1).strip().lower() if type_match else None
                    if identified_type:
                        for prop_candidate in available_properties:
                            candidate_type = prop_candidate.get('subject',{}).get('structure_type','').lower()
                            if identified_type == candidate_type:
                                if not any(rec.get('orderID') == prop_candidate.get('orderID') for rec in recommendations_data):
                                    matched_prop = prop_candidate
                                    break # take the first type match

                if matched_prop:
                    # Add similarity score (placeholder, as LLM doesn't give explicit score here)
                    # Could be based on position in LLM output, or a post-hoc similarity calculation
                    matched_prop_copy = matched_prop.copy()
                    matched_prop_copy['similarity_score'] = 1.0 - (len(recommendations_data) * 0.1) # Higher score for earlier items
                    recommendations_data.append(matched_prop_copy)

        except Exception as e:
            logger.error(f"Error during recommendation generation: {str(e)}")
            # Fallback strategy if LLM generation fails or parsing is problematic
            
        # Fallback if LLM yields too few results
        if len(recommendations_data) < top_n:
            logger.warning(f"LLM generated {len(recommendations_data)} comps, attempting fallback for {top_n - len(recommendations_data)} more.")
            # Basic fallback: properties of the same type, not already recommended
            
            query_prop_type = query_property.get('subject', {}).get('structure_type', '').lower()
            
            already_recommended_ids = {rec.get('orderID') for rec in recommendations_data}
            
            fallback_candidates = [
                p for p in available_properties 
                if p.get('subject', {}).get('structure_type', '').lower() == query_prop_type 
                and p.get('orderID') not in already_recommended_ids
            ]
            
            # Simple distance or other heuristic could be used to sort fallbacks
            # For now, just take them in order
            for fb_prop in fallback_candidates:
                if len(recommendations_data) >= top_n: break
                fb_prop_copy = fb_prop.copy()
                fb_prop_copy['similarity_score'] = 0.5 # Indicate lower confidence for fallback
                recommendations_data.append(fb_prop_copy)

        return recommendations_data[:top_n]

    def generate_explanation(self, 
                           query_property: Dict[str, Any], 
                           recommendation: Dict[str, Any],
                           detailed: bool = False) -> str:
        """
        Generate an explanation for why a property was recommended using the fine-tuned LLAMA model.
        """
        if not self.is_fit or self.model is None or self.tokenizer is None:
            return "Model has not been fine-tuned or initialized. Cannot generate explanation."
            
        query_text = self._property_to_text(query_property)
        rec_text = self._property_to_text(recommendation)
        
        alpaca_prompt_inference = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
        instruction = (
            f"Explain why the 'Recommended Property' is a good comparable for the 'Query Property' for valuation purposes. "
            f"Provide a {'detailed' if detailed else 'concise'} explanation focusing on key similarities."
        )
        input_text = (
            f"Query Property:\n{query_text}\n\nRecommended Property:\n{rec_text}"
        )
        prompt = alpaca_prompt_inference.format(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length - (256 if detailed else 128)).to(self.device)
        
        logger.info("Generating explanation with fine-tuned LLAMA model...")
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256 if detailed else 128,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            explanation = generated_text.split("### Response:")[-1].strip()
            return explanation if explanation else "Could not generate a specific explanation."
        except Exception as e:
            logger.error(f"Error during explanation generation: {str(e)}")
            return "An error occurred while generating the explanation."

if __name__ == '__main__':
    # Example Usage (Illustrative - requires actual data)
    logger.info("Starting LlamaPropertyRecommender example usage...")

    # Mock data for demonstration
    mock_properties_data = [
        {
            "orderID": "prop1", 
            "subject": {
                "address": "123 Main St, Anytown, ON A1A1A1", "structure_type": "Detached", 
                "style": "Bungalow", "year_built": "1990", "gla": "1500", "lot_size_sf": "5000",
                "condition": "Average", "room_count": "6", "num_beds": "3", "num_baths": "2",
                "municipality_district": "Anytown Central"
            },
            "comps": [
                {"address": "125 Main St", "city_province": "Anytown, ON", "prop_type": "Detached", "stories": "Bungalow", "sale_date": "2023-01-15", "sale_price": "500000", "distance_to_subject": "0.1km", "year_built": "1992", "gla": "1550"},
                {"address": "45 Suburb Ave", "city_province": "Anytown, ON", "prop_type": "Detached", "stories": "2-Storey", "sale_date": "2023-02-20", "sale_price": "550000", "distance_to_subject": "1.5km", "year_built": "1995", "gla": "1800"},
            ]
        },
        {
            "orderID": "prop2", 
            "subject": {
                "address": "789 Oak Rd, Otherville, BC B2B2B2", "structure_type": "Townhouse", 
                "style": "2-Storey", "year_built": "2005", "gla": "1200", "lot_size_sf": "2000",
                "condition": "Good", "room_count": "5", "num_beds": "2", "num_baths": "1.5",
                "municipality_district": "Otherville West"
            },
            "comps": [
                {"address": "791 Oak Rd", "city_province": "Otherville, BC", "prop_type": "Townhouse", "stories": "2-Storey", "sale_date": "2022-12-01", "sale_price": "400000", "distance_to_subject": "0.05km", "year_built": "2006", "gla": "1210"},
            ]
        },
        # Add more mock properties for more robust testing
        {
            "orderID": "prop3", 
            "subject": {
                "address": "321 Pine Ln, Anytown, ON A1A2A2", "structure_type": "Detached", 
                "style": "Bungalow", "year_built": "1988", "gla": "1450", "lot_size_sf": "5200",
                "condition": "Good", "room_count": "6", "num_beds": "3", "num_baths": "2",
                 "municipality_district": "Anytown Central"
            },
             "comps": [] # Property with no comps initially
        },
         {
            "orderID": "prop4_query", 
            "subject": {
                "address": "10 Downing Street, Anytown, ON A1A1A1", "structure_type": "Detached", 
                "style": "Bungalow", "year_built": "1991", "gla": "1520", "lot_size_sf": "5100",
                "condition": "Average", "room_count": "6", "num_beds": "3", "num_baths": "2",
                 "municipality_district": "Anytown Central"
            },
            "comps": [] 
        }
    ]

    # --- Configuration ---
    # Ensure CUDA is available or it will be very slow
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. LLAMA operations will be extremely slow on CPU.")
        # return # Exit if no CUDA for a real run

    cache_directory = "./llama_cache" # Make sure this directory exists or can be created
    Path(cache_directory).mkdir(parents=True, exist_ok=True)

    recommender = LlamaPropertyRecommender(
        model_name="unsloth/llama-2-7b-hf-unsloth", # Use a small, fast model for testing if needed
        cache_dir=cache_directory,
        max_length=1024 # Adjust based on your VRAM and data
    )

    # --- Fit the model ---
    # In a real scenario, you would load your full dataset here
    try:
        logger.info("Attempting to fit the model...")
        recommender.fit(
            properties=mock_properties_data, 
            num_epochs=1, # Keep epochs low for quick testing
            batch_size=1, # Small batch size for testing on limited VRAM
            lr=5e-5 # Common learning rate for fine-tuning
        )
    except Exception as e:
        logger.error(f"Exception during fit: {e}", exc_info=True)
        # Depending on the error, you might want to stop or try to load a pre-trained model

    # --- Get Recommendations ---
    if recommender.is_fit:
        logger.info("Model is fit. Attempting to get recommendations...")
        query_property_example = mock_properties_data[3] # Use prop4_query
        
        recommendations = recommender.recommend(
            query_property=query_property_example, 
            top_n=2,
            filter_by_location=True, # Test location filtering
            location_params={"city": "Anytown", "province": "ON"} 
        )
        
        if recommendations:
            print(f"\n--- Recommendations for Query Property ID: {query_property_example['orderID']} ---")
            for i, rec in enumerate(recommendations):
                print(f"\nRecommendation {i+1}:")
                print(f"  Property ID: {rec.get('orderID')}")
                print(f"  Address: {rec.get('subject', {}).get('address')}")
                print(f"  Type: {rec.get('subject', {}).get('structure_type')}")
                print(f"  Similarity Score (Placeholder): {rec.get('similarity_score', 'N/A')}")

                # --- Generate Explanation ---
                explanation = recommender.generate_explanation(query_property_example, rec, detailed=False)
                print(f"  Explanation: {explanation}")
        else:
            print(f"\nNo recommendations found for Property ID: {query_property_example['orderID']}")
    else:
        logger.warning("Model was not fit successfully. Skipping recommendation and explanation.")
    
    logger.info("LlamaPropertyRecommender example usage finished.") 