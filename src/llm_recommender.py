import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from pathlib import Path


class LLMPropertyRecommender:
    """
    LLM-based property recommendation system using sentence embeddings.
    """
    
    def __init__(self, 
                 model_name: str = 'paraphrase-MiniLM-L6-v2',
                 cache_embeddings: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize the LLM-based property recommender.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_embeddings: Whether to cache property embeddings
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.property_embeddings = None
        self.property_data = None
        self.is_fit = False
        self.location_data = {}  # To store location information for each property
        self.feature_weights = {  # Default weights for property features in text representation
            'structure_type': 1.5,
            'style': 1.5,
            'year_built': 1.2,
            'gla': 1.2,
            'lot_size': 1.2,
            'condition': 1.2,
            'room_count': 1.1,
            'bed_count': 1.1,
            'bath_count': 1.1,
            'address': 0.7,  # Lower weight for address to focus more on property features
            'basement': 1.0,
            'construction': 1.0,
            'heating': 0.9,
            'cooling': 0.9
        }
        
    def _property_to_text(self, prop: Dict[str, Any]) -> str:
        """
        Convert a property dictionary to a text description for the LLM.
        
        Args:
            prop: Property dictionary
            
        Returns:
            Text description of the property
        """
        # Extract subject information
        subject = prop.get('subject', {})
        
        # Extract key features with weights
        features = []
        
        # Process address and location
        address = subject.get('address', 'unknown address')
        features.append(f"Property at {address}.")
        
        # Add municipality/district
        municipality = subject.get('municipality_district', '')
        if municipality:
            features.append(f"Located in {municipality}.")
        
        # Process key features
        structure_type = subject.get('structure_type', '')
        if structure_type:
            weight = self.feature_weights.get('structure_type', 1.0)
            features.append(f"Type: {structure_type}." * int(weight))
            
        style = subject.get('style', '')
        if style:
            weight = self.feature_weights.get('style', 1.0)
            features.append(f"Style: {style}." * int(weight))
            
        year_built = subject.get('year_built', '')
        if year_built:
            weight = self.feature_weights.get('year_built', 1.0)
            features.append(f"Year built: {year_built}." * int(weight))
            
        construction = subject.get('construction', '')
        if construction:
            weight = self.feature_weights.get('construction', 1.0)
            features.append(f"Construction: {construction}." * int(weight))
            
        lot_size = subject.get('lot_size_sf', '')
        if lot_size and lot_size != 'n/a':
            weight = self.feature_weights.get('lot_size', 1.0)
            features.append(f"Lot size: {lot_size} sq ft." * int(weight))
            
        gla = subject.get('gla', '')
        if gla:
            weight = self.feature_weights.get('gla', 1.0)
            features.append(f"Gross living area: {gla} sq ft." * int(weight))
            
        condition = subject.get('condition', '')
        if condition:
            weight = self.feature_weights.get('condition', 1.0)
            features.append(f"Condition: {condition}." * int(weight))
        
        # Room information
        rooms = subject.get('room_count', '')
        if rooms:
            weight = self.feature_weights.get('room_count', 1.0)
            features.append(f"Room count: {rooms}." * int(weight))
            
        beds = subject.get('num_beds', '')
        if beds:
            weight = self.feature_weights.get('bed_count', 1.0)
            features.append(f"Bedrooms: {beds}." * int(weight))
            
        baths = subject.get('num_baths', '')
        if baths:
            weight = self.feature_weights.get('bath_count', 1.0)
            features.append(f"Bathrooms: {baths}." * int(weight))
        
        # Building features
        basement = subject.get('basement', '')
        if basement:
            weight = self.feature_weights.get('basement', 1.0)
            features.append(f"Basement: {basement}." * int(weight))
            
        heating = subject.get('heating', '')
        if heating:
            weight = self.feature_weights.get('heating', 1.0)
            features.append(f"Heating: {heating}." * int(weight))
            
        cooling = subject.get('cooling', '')
        if cooling:
            weight = self.feature_weights.get('cooling', 1.0)
            features.append(f"Cooling: {cooling}." * int(weight))
        
        # Create final description
        description = " ".join(features)
        
        return description
    
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
    
    def _properties_to_texts(self, properties: List[Dict[str, Any]]) -> List[str]:
        """
        Convert a list of property dictionaries to text descriptions.
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            List of text descriptions
        """
        return [self._property_to_text(prop) for prop in properties]
    
    def fit(self, properties: List[Dict[str, Any]]) -> None:
        """
        Fit the LLM-based model to the property data.
        
        Args:
            properties: List of property dictionaries
        """
        self.property_data = properties
        
        # Extract location data
        self.location_data = self._extract_location_data(properties)
        
        # Check if cached embeddings exist
        if self.cache_embeddings and self.cache_dir and self._load_cached_embeddings():
            logging.info("Loaded embeddings from cache.")
            self.is_fit = True
            return
        
        # Convert properties to text descriptions
        property_texts = self._properties_to_texts(properties)
        
        # Generate embeddings for all properties
        logging.info(f"Generating embeddings for {len(property_texts)} properties...")
        self.property_embeddings = self.model.encode(property_texts, show_progress_bar=True)
        logging.info("Embeddings generated successfully.")
        
        # Cache embeddings if enabled
        if self.cache_embeddings and self.cache_dir:
            self._cache_embeddings()
            
        self.is_fit = True
        
    def _cache_embeddings(self) -> bool:
        """
        Cache property embeddings to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_dir:
            return False
            
        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            cache_path = self.cache_dir / f"embeddings_{self.model_name.replace('-', '_')}.npy"
            np.save(cache_path, self.property_embeddings)
            
            # Save property IDs for verification
            ids_path = self.cache_dir / "property_ids.json"
            with open(ids_path, 'w') as f:
                property_ids = [prop.get('orderID') for prop in self.property_data]
                json.dump(property_ids, f)
                
            return True
        except Exception as e:
            logging.error(f"Error caching embeddings: {e}")
            return False
    
    def _load_cached_embeddings(self) -> bool:
        """
        Load cached property embeddings from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_dir:
            return False
            
        try:
            # Check if cache files exist
            cache_path = self.cache_dir / f"embeddings_{self.model_name.replace('-', '_')}.npy"
            ids_path = self.cache_dir / "property_ids.json"
            
            if not cache_path.exists() or not ids_path.exists():
                return False
                
            # Load property IDs and verify match
            with open(ids_path, 'r') as f:
                cached_ids = json.load(f)
                
            current_ids = [prop.get('orderID') for prop in self.property_data]
            
            if len(cached_ids) != len(current_ids) or set(cached_ids) != set(current_ids):
                logging.warning("Cached property IDs don't match current data. Regenerating embeddings.")
                return False
                
            # Load embeddings
            self.property_embeddings = np.load(cache_path)
            
            return True
        except Exception as e:
            logging.error(f"Error loading cached embeddings: {e}")
            return False
    
    def recommend(self, 
                 query_property: Dict[str, Any], 
                 top_n: int = 5,
                 filter_by_location: bool = True,
                 location_params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend similar properties based on a query property.
        
        Args:
            query_property: Property to find similar properties for
            top_n: Number of recommendations to return
            filter_by_location: Whether to filter results by location
            location_params: Location parameters for filtering
            
        Returns:
            List of recommended properties sorted by similarity
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
            
        # Convert query property to text
        query_text = self._property_to_text(query_property)
        
        # Generate embedding for query property
        query_embedding = self.model.encode([query_text])[0]
        
        # Compute similarities with all property embeddings
        similarities = cosine_similarity([query_embedding], self.property_embeddings)[0]
        
        # Apply location filtering if enabled
        if filter_by_location and location_params:
            filtered_indices = []
            # Get order IDs of properties
            property_ids = [prop.get('orderID') for prop in self.property_data]
            
            # Apply location filtering
            for i, order_id in enumerate(property_ids):
                if self._matches_location_criteria(order_id, location_params):
                    filtered_indices.append(i)
                    
            # Get similarities for filtered properties
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            
            # Sort by similarity
            sorted_similarities = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)
            
            # Get top_n indices
            top_indices = [idx for idx, _ in sorted_similarities[:top_n]]
        else:
            # Get indices of top_n similar properties
            top_indices = np.argsort(-similarities)[:top_n]
        
        # Get the recommended properties
        recommendations = []
        for idx in top_indices:
            if idx < len(self.property_data):  # Safety check
                rec = self.property_data[idx].copy()
                rec['similarity_score'] = float(similarities[idx])
                recommendations.append(rec)
            
        return recommendations
    
    def _matches_location_criteria(self, 
                                  order_id: str, 
                                  location_params: Dict[str, str]) -> bool:
        """
        Check if a property matches the specified location criteria.
        
        Args:
            order_id: Order ID of the property
            location_params: Location parameters for filtering
            
        Returns:
            True if property matches location criteria, False otherwise
        """
        if order_id not in self.location_data:
            return False
            
        location_info = self.location_data[order_id]
        
        # Check each criterion if specified
        if 'city' in location_params and location_params['city']:
            if location_params['city'].lower() not in location_info.get('city', '').lower():
                return False
                
        if 'province' in location_params and location_params['province']:
            if location_params['province'].upper() != location_info.get('province', '').upper():
                return False
                
        if 'postal_code' in location_params and location_params['postal_code']:
            # Match first 3 characters of postal code (Forward Sortation Area)
            if not location_info.get('postal_code', '') or location_params['postal_code'][:3].upper() != location_info.get('postal_code', '')[:3].upper():
                return False
                
        if 'municipality' in location_params and location_params['municipality']:
            if location_params['municipality'].lower() not in location_info.get('municipality', '').lower():
                return False
                
        return True
    
    def generate_explanation(self, 
                            query_property: Dict[str, Any], 
                            recommendation: Dict[str, Any],
                            detailed: bool = False) -> str:
        """
        Generate a natural language explanation for why a property was recommended.
        
        Args:
            query_property: The query property
            recommendation: The recommended property
            detailed: Whether to generate a detailed explanation
            
        Returns:
            Explanation string
        """
        # Extract subject information
        query_subject = query_property.get('subject', {})
        rec_subject = recommendation.get('subject', {})
        
        # Create a basic explanation
        if not detailed:
            explanation = (
                f"This property at {rec_subject.get('address', 'unknown')} was recommended because "
                f"it is similar to your query property in terms of structure type "
                f"({rec_subject.get('structure_type', 'unknown')}), style "
                f"({rec_subject.get('style', 'unknown')}), and year built "
                f"({rec_subject.get('year_built', 'unknown')})."
            )
            return explanation
        
        # Create a more detailed explanation
        similarities = []
        differences = []
        
        # Compare important features with custom weighting
        features_to_compare = [
            ('structure_type', 'structure type', 1.5),
            ('style', 'style', 1.5),
            ('year_built', 'year built', 1.2),
            ('construction', 'construction', 1.0),
            ('lot_size_sf', 'lot size', 1.2),
            ('gla', 'gross living area', 1.2),
            ('condition', 'condition', 1.2),
            ('room_count', 'room count', 1.0),
            ('num_beds', 'bedrooms', 1.1),
            ('num_baths', 'bathrooms', 1.1),
            ('basement', 'basement', 1.0),
            ('heating', 'heating', 0.9),
            ('cooling', 'cooling', 0.9)
        ]
        
        # Features with exact matches get a boost in importance
        for key, desc, weight in features_to_compare:
            query_val = query_subject.get(key, 'unknown')
            rec_val = rec_subject.get(key, 'unknown')
            
            if query_val == rec_val and query_val != 'unknown':
                importance = f"important" if weight > 1.2 else ""
                importance = f"very important" if weight > 1.4 else importance
                importance = f" ({importance})" if importance else ""
                similarities.append(f"both have the same {desc}{importance}: {query_val}")
            elif query_val != 'unknown' and rec_val != 'unknown':
                differences.append(f"different {desc}: query property has {query_val}, recommended property has {rec_val}")
        
        # Check for location similarities
        query_location = query_subject.get('municipality_district', '')
        rec_location = rec_subject.get('municipality_district', '')
        
        if query_location and rec_location:
            if query_location.lower() == rec_location.lower():
                similarities.append(f"both located in the same area: {query_location}")
            else:
                differences.append(f"located in different areas: query in {query_location}, recommendation in {rec_location}")
        
        # Build the explanation
        explanation = f"This property at {rec_subject.get('address', 'unknown')} was recommended because:\n"
        
        if similarities:
            explanation += "Similarities:\n- " + "\n- ".join(similarities) + "\n"
            
        if differences:
            explanation += "Differences (but still considered similar overall):\n- " + "\n- ".join(differences)
            
        explanation += f"\nOverall similarity score: {recommendation.get('similarity_score', 0):.2f}"
        
        return explanation
    
    def save_embeddings(self, filepath: str) -> None:
        """
        Save property embeddings to a file.
        
        Args:
            filepath: Path to save the embeddings
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
            
        # Save the embeddings
        np.save(filepath, self.property_embeddings)
        
    def load_embeddings(self, 
                        filepath: str, 
                        properties: List[Dict[str, Any]]) -> None:
        """
        Load property embeddings from a file.
        
        Args:
            filepath: Path to load the embeddings from
            properties: List of property dictionaries
        """
        # Load the embeddings
        self.property_embeddings = np.load(filepath)
        self.property_data = properties
        
        # Extract location data
        self.location_data = self._extract_location_data(properties)
        
        self.is_fit = True
    
    def update_feature_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update feature weights for text representation.
        
        Args:
            new_weights: Dictionary of feature weights to update
        """
        self.feature_weights.update(new_weights)
        
        # Clear embeddings to force recomputation with new weights
        if self.is_fit:
            logging.info("Feature weights updated. Embeddings will be recomputed on next fit.")
            self.property_embeddings = None
            self.is_fit = False
        
    def fine_tune(self, 
                 query_properties: List[Dict[str, Any]], 
                 similar_properties: List[List[Dict[str, Any]]],
                 learning_rate: float = 0.001,
                 epochs: int = 5) -> None:
        """
        Fine-tune the LLM for better property similarity matching.
        
        Note: This is a placeholder method. Actual fine-tuning would require 
        integrating with libraries like HuggingFace Transformers for model training.
        
        Args:
            query_properties: List of query properties
            similar_properties: Lists of similar properties for each query
            learning_rate: Learning rate for fine-tuning
            epochs: Number of epochs for fine-tuning
        """
        logging.warning(
            "Fine-tuning is not implemented in this version. "
            "To fine-tune an LLM, you would need to use HuggingFace Transformers "
            "or similar libraries with appropriate training data."
        )
        
        # Auto-tune feature weights based on similar properties
        if query_properties and similar_properties:
            self._auto_tune_feature_weights(query_properties, similar_properties)
    
    def _auto_tune_feature_weights(self, 
                                 query_properties: List[Dict[str, Any]], 
                                 similar_properties: List[List[Dict[str, Any]]]) -> None:
        """
        Automatically tune feature weights based on provided similar properties.
        
        Args:
            query_properties: List of query properties
            similar_properties: Lists of similar properties for each query
        """
        logging.info("Auto-tuning feature weights based on provided similar properties...")
        
        # Feature importance counters
        feature_importance = {key: 0.0 for key in self.feature_weights.keys()}
        feature_count = {key: 0 for key in self.feature_weights.keys()}
        
        # Analyze feature matches in similar properties
        for i, query_prop in enumerate(query_properties):
            if i >= len(similar_properties):
                continue
                
            query_subject = query_prop.get('subject', {})
            
            for similar_prop in similar_properties[i]:
                similar_subject = similar_prop.get('subject', {})
                
                # For each feature, check if it matches
                for feature in feature_importance.keys():
                    subject_feature = feature
                    
                    # Map feature names to subject keys if needed
                    if feature == 'bed_count':
                        subject_feature = 'num_beds'
                    elif feature == 'bath_count':
                        subject_feature = 'num_baths'
                    elif feature == 'lot_size':
                        subject_feature = 'lot_size_sf'
                    
                    # Skip address
                    if feature == 'address':
                        continue
                        
                    query_val = query_subject.get(subject_feature, None)
                    similar_val = similar_subject.get(subject_feature, None)
                    
                    # If both values exist and match, increase importance
                    if query_val and similar_val and query_val == similar_val:
                        feature_importance[feature] += 1.0
                        
                    feature_count[feature] += 1
        
        # Calculate new weights
        new_weights = {}
        for feature, importance in feature_importance.items():
            if feature_count[feature] > 0:
                # Formula: base weight (1.0) + bonus based on match frequency
                match_frequency = importance / feature_count[feature]
                new_weight = 1.0 + match_frequency
                
                # Cap weights between 0.5 and 2.0
                new_weight = max(0.5, min(2.0, new_weight))
                
                new_weights[feature] = new_weight
        
        # Update weights if values were calculated
        if new_weights:
            self.update_feature_weights(new_weights)
            logging.info(f"Auto-tuned feature weights: {new_weights}")
        else:
            logging.warning("Could not auto-tune feature weights from provided data.") 