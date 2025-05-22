import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any, Tuple, Optional
import re


class AppraisalDataLoader:
    """
    Class to load and preprocess appraisal data for property recommendation.
    """
    
    def __init__(self, data_path: str = "src/data/appraisals_dataset.json"):
        """
        Initialize the data loader with the path to the appraisal dataset.
        
        Args:
            data_path: Path to the appraisal dataset JSON file
        """
        self.data_path = Path(data_path)
        self.appraisals = None
        self.appraisals_df = None
        self.properties_df = None  # To store flattened properties data
        self.comps_df = None       # To store flattened comps data
        self.features = None
        self.location_data = None  # To store location data
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load appraisal data from JSON file.
        
        Returns:
            List of appraisal data dictionaries
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        self.appraisals = data.get('appraisals', [])
        return self.appraisals
    
    def extract_location_from_address(self, address: str) -> Dict[str, str]:
        """
        Extract city, province, and postal code from address string.
        
        Args:
            address: Full address string
            
        Returns:
            Dictionary with extracted location information
        """
        location_info = {
            'city': '',
            'province': '',
            'postal_code': ''
        }
        
        # Regular expression to find postal code (Canadian format: A1A 1A1)
        postal_pattern = r'[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d'
        postal_match = re.search(postal_pattern, address)
        if postal_match:
            location_info['postal_code'] = postal_match.group(0)
        
        # Extract province (assuming standard Canadian province abbreviations)
        provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
        for province in provinces:
            if f" {province} " in f" {address} ":
                location_info['province'] = province
                break
        
        # Extract city (simple approach - this would need refinement in a real application)
        if location_info['province'] and location_info['province'] in address:
            parts = address.split(location_info['province'])[0].strip().split()
            if parts:
                location_info['city'] = parts[-1]
                
        # Fallback to municipality_district if city couldn't be extracted
        return location_info
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert appraisal data to a pandas DataFrame for easier manipulation.
        Also extracts properties and comps data into separate DataFrames.
        
        Returns:
            DataFrame containing appraisal data
        """
        if self.appraisals is None:
            self.load_data()
            
        # Flatten the nested structure of appraisals
        flattened_data = []
        all_properties = []
        all_comps = []
        
        for appraisal in self.appraisals:
            flat_appraisal = {}
            flat_appraisal['orderID'] = appraisal.get('orderID')
            
            # Flatten subject properties
            subject = appraisal.get('subject', {})
            for key, value in subject.items():
                flat_appraisal[f'subject_{key}'] = value
                
            # Extract location information
            address = subject.get('address', '')
            municipality = subject.get('municipality_district', '')
            
            location_info = self.extract_location_from_address(address)
            flat_appraisal['subject_city'] = location_info['city'] or municipality
            flat_appraisal['subject_province'] = location_info['province']
            flat_appraisal['subject_postal_code'] = location_info['postal_code']
            
            # Add to flattened data
            flattened_data.append(flat_appraisal)
            
            # Process properties if they exist
            order_id = appraisal.get('orderID')
            properties = appraisal.get('properties', [])
            for prop in properties:
                prop['orderID'] = order_id
                all_properties.append(prop)
                
            # Process comps if they exist
            comps = appraisal.get('comps', [])
            for comp in comps:
                comp['orderID'] = order_id
                all_comps.append(comp)
            
        # Create DataFrames
        self.appraisals_df = pd.DataFrame(flattened_data)
        
        # Create properties DataFrame if properties exist
        if all_properties:
            self.properties_df = pd.DataFrame(all_properties)
            
        # Create comps DataFrame if comps exist
        if all_comps:
            self.comps_df = pd.DataFrame(all_comps)
            
        return self.appraisals_df
    
    def preprocess_features(self) -> np.ndarray:
        """
        Preprocess and extract features from appraisal data for model input.
        
        Returns:
            NumPy array of preprocessed features
        """
        if self.appraisals_df is None:
            self.to_dataframe()
            
        # Extract relevant features
        # Numerical features from subject properties
        numerical_features = [
            'subject_lot_size_sf', 
            'subject_year_built', 
            'subject_effective_age',
            'subject_gla',
            'subject_main_lvl_area',
            'subject_second_lvl_area',
            'subject_basement_area',
            'subject_room_count',
            'subject_num_beds'
        ]
        
        # Categorical features from subject properties
        categorical_features = [
            'subject_structure_type', 
            'subject_style', 
            'subject_construction',
            'subject_basement',
            'subject_heating',
            'subject_cooling',
            'subject_condition',
            'subject_city',
            'subject_province'
        ]
        
        # Handle numerical features
        num_data = self.appraisals_df[numerical_features].copy()
        
        # Convert non-numeric values to NaN
        for col in num_data.columns:
            num_data[col] = pd.to_numeric(num_data[col], errors='coerce')
            
        # Fill NaN values with column means
        num_data = num_data.fillna(num_data.mean())
        
        # Handle categorical features using one-hot encoding
        cat_data = pd.get_dummies(self.appraisals_df[categorical_features], drop_first=True)
        
        # Combine all features
        features_df = pd.concat([num_data, cat_data], axis=1)
        self.features = features_df.values
        
        # Store feature names for later use
        self.feature_names = list(features_df.columns)
        
        return self.features
    
    def preprocess_location_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract and preprocess location data for filtering.
        
        Returns:
            Dictionary mapping orderIDs to location information
        """
        if self.appraisals_df is None:
            self.to_dataframe()
            
        location_data = {}
        
        for _, row in self.appraisals_df.iterrows():
            order_id = row.get('orderID')
            location_data[order_id] = {
                'city': row.get('subject_city', ''),
                'province': row.get('subject_province', ''),
                'postal_code': row.get('subject_postal_code', ''),
                'municipality': row.get('subject_municipality_district', '')
            }
            
        self.location_data = location_data
        return location_data
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if self.features is None:
            self.preprocess_features()
            
        return self.feature_names
    
    def filter_by_location(self, 
                          city: Optional[str] = None, 
                          province: Optional[str] = None,
                          postal_code: Optional[str] = None,
                          municipality: Optional[str] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Filter properties by location.
        
        Args:
            city: City to filter by
            province: Province to filter by
            postal_code: Postal code to filter by
            municipality: Municipality/district to filter by
            
        Returns:
            Tuple of (filtered appraisals, filtered features)
        """
        if self.appraisals_df is None or self.features is None:
            self.to_dataframe()
            self.preprocess_features()
            
        if self.location_data is None:
            self.preprocess_location_data()
            
        # Start with all indices
        valid_indices = set(range(len(self.appraisals_df)))
        
        # Apply each filter if provided
        if city:
            city_indices = {i for i, row in enumerate(self.appraisals_df.iterrows()) 
                           if city.lower() in str(row[1].get('subject_city', '')).lower()}
            valid_indices &= city_indices
            
        if province:
            province_indices = {i for i, row in enumerate(self.appraisals_df.iterrows())
                               if province.upper() == str(row[1].get('subject_province', '')).upper()}
            valid_indices &= province_indices
            
        if postal_code:
            # Match first 3 characters of postal code (FSA)
            postal_indices = {i for i, row in enumerate(self.appraisals_df.iterrows())
                             if postal_code[:3].upper() == str(row[1].get('subject_postal_code', ''))[:3].upper()}
            valid_indices &= postal_indices
            
        if municipality:
            municipality_indices = {i for i, row in enumerate(self.appraisals_df.iterrows())
                                  if municipality.lower() in str(row[1].get('subject_municipality_district', '')).lower()}
            valid_indices &= municipality_indices
            
        # Convert indices to list and sort
        valid_indices = sorted(list(valid_indices))
        
        # Get filtered appraisals and features
        filtered_appraisals = [self.appraisals[i] for i in valid_indices]
        filtered_features = self.features[valid_indices]
        
        return filtered_appraisals, filtered_features
    
    def get_properties_for_appraisal(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get properties for a specific appraisal.
        
        Args:
            order_id: OrderID of the appraisal
            
        Returns:
            List of properties for the appraisal
        """
        if self.properties_df is None:
            self.to_dataframe()
            
        if self.properties_df is None:  # If still None, no properties exist
            return []
            
        properties = self.properties_df[self.properties_df['orderID'] == order_id].to_dict('records')
        return properties
    
    def get_comps_for_appraisal(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get comps for a specific appraisal.
        
        Args:
            order_id: OrderID of the appraisal
            
        Returns:
            List of comps for the appraisal
        """
        if self.comps_df is None:
            self.to_dataframe()
            
        if self.comps_df is None:  # If still None, no comps exist
            return []
            
        comps = self.comps_df[self.comps_df['orderID'] == order_id].to_dict('records')
        return comps 