import json
import re
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


N_COMPS_TO_FIND = 3
NUMERICAL_FEATURES = ['gla', 'age', 'lot_size_sf', 'bedrooms', 'total_baths', 'distance_to_subject_km']
CATEGORICAL_FEATURES = ['structure_type', 'style', 'condition']

# --- Helper Functions ---

def clean_string(text: Optional[str]) -> Optional[str]:
    return text.strip() if text else None

def parse_sq_ft(value: Optional[Any]) -> Optional[float]: # Changed to float to handle potential non-integer SQFT
    if value is None or str(value).lower() == "n/a":
        return None
    match = re.search(r"(\\d+\\.?\\d*|\\d+)", str(value).replace(",", "")) # Allows for decimals
    return float(match.group(1)) if match else None

def parse_price(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError:
        return None

def parse_bath_count(bath_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parses bath strings like "1:1" (full:half) or "2" (full only) or "2:0" """
    if not bath_str:
        return None, None
    parts = str(bath_str).split(':')
    full_baths = None
    half_baths = None
    try:
        if len(parts) == 1 and parts[0].strip():
            full_baths = int(parts[0].strip())
            half_baths = 0 
        elif len(parts) == 2:
            full_baths = int(parts[0].strip()) if parts[0].strip() else 0
            half_baths = int(parts[1].strip()) if parts[1].strip() else 0
    except ValueError:
        return None, None # Error in parsing
    return full_baths, half_baths

def calculate_age(year_built: Optional[Any], reference_year: Optional[int]) -> Optional[int]:
    if year_built and reference_year:
        try:
            # Handle cases like "1976 (Assumed)" or other non-purely numeric strings
            year_built_cleaned = re.search(r"(\\d{4})", str(year_built))
            if year_built_cleaned:
                return reference_year - int(year_built_cleaned.group(1))
            return None
        except ValueError:
            return None
    return None

def parse_effective_date_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        dt_obj = datetime.strptime(date_str, "%b/%d/%Y")
        return dt_obj.year
    except ValueError:
        try: # Attempt another common format like YYYY-MM-DD
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return dt_obj.year
        except ValueError:
            return None

def haversine_distance(lat1: Optional[float], lon1: Optional[float], lat2: Optional[float], lon2: Optional[float]) -> Optional[float]:
    """Calculate distance in KM between two lat/lon points."""
    if None in [lat1, lon1, lat2, lon2]:
        return None
    R = 6371.0  # Radius of Earth in kilometers

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# --- Standardization Functions ---

def standardize_property_features(prop_data: Dict[str, Any], 
                                  data_type: str, # "subject", "comp", "property_listing"
                                  reference_year: Optional[int] = None,
                                  subject_lat: Optional[float] = None, 
                                  subject_lon: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not prop_data:
        return None

    std = {"data_type": data_type, "original_data": prop_data} # Keep original for reference
    
    std['id'] = prop_data.get('orderID', prop_data.get('id', clean_string(prop_data.get('address'))))
    std['address_full'] = clean_string(prop_data.get('address'))
    
    if data_type == 'comp':
        city_province_str = prop_data.get('city_province', '')
        city_parts = city_province_str.split(' ON ')
        std['city'] = clean_string(city_parts[0]) if len(city_parts) > 0 else None
        std['province'] = "ON" # Assuming Ontario from "ON"
        std['postal_code'] = clean_string(city_province_str.split(' ')[-1]) if ' ' in city_province_str else None
    else:
        std['city'] = clean_string(prop_data.get('city'))
        std['province'] = clean_string(prop_data.get('province'))
        std['postal_code'] = clean_string(prop_data.get('postal_code'))


    std['gla'] = parse_sq_ft(prop_data.get('gla'))
    
    year_built_raw = prop_data.get('year_built')
    std['year_built'] = int(re.search(r"(\\d{4})", str(year_built_raw)).group(1)) if year_built_raw and re.search(r"(\\d{4})", str(year_built_raw)) else None
    
    if data_type == 'subject':
        age_raw = prop_data.get('effective_age', prop_data.get('subject_age'))
        std['age'] = int(age_raw) if age_raw and str(age_raw).isdigit() else calculate_age(std['year_built'], reference_year)
    elif data_type == 'comp':
        age_raw = prop_data.get('age')
        std['age'] = int(age_raw) if age_raw and str(age_raw).isdigit() else calculate_age(std['year_built'], reference_year)
    else: # property_listing
        std['age'] = calculate_age(std['year_built'], reference_year)

    std['lot_size_sf'] = parse_sq_ft(prop_data.get('lot_size_sf', prop_data.get('lot_size')))

    if data_type == 'subject':
        std['bedrooms'] = int(prop_data['num_beds']) if prop_data.get('num_beds', '').isdigit() else None
    elif data_type == 'comp':
        std['bedrooms'] = int(prop_data['bed_count']) if prop_data.get('bed_count', '').isdigit() else None
    else: # property_listing
        std['bedrooms'] = prop_data.get('bedrooms') # Assumed int or None
        if std['bedrooms'] is not None:
            try:
                std['bedrooms'] = int(std['bedrooms'])
            except ValueError:
                std['bedrooms'] = None


    bath_source_str = prop_data.get('num_baths') if data_type == 'subject' else \
                      prop_data.get('bath_count') if data_type == 'comp' else None
    
    if bath_source_str: # Subject or Comp with specific format
        fb, hb = parse_bath_count(bath_source_str)
    else: # Property listing or if specific format fails
        fb = prop_data.get('full_baths')
        hb = prop_data.get('half_baths')
        try: # Ensure they are integers if present
            fb = int(fb) if fb is not None else None
            hb = int(hb) if hb is not None else None
        except ValueError: # If conversion fails, set to None
            fb, hb = None, None
            
    std['full_baths'], std['half_baths'] = fb, hb
    std['total_baths'] = (std.get('full_baths') or 0) + 0.5 * (std.get('half_baths') or 0)

    if data_type == 'subject':
        raw_struct = prop_data.get('structure_type')
    elif data_type == 'comp':
        raw_struct = prop_data.get('prop_type')
    else: # property_listing
        raw_struct = prop_data.get('structure_type', prop_data.get('property_sub_type'))
    
    std['structure_type'] = clean_string(raw_struct.split(',')[0] if raw_struct and ',' in raw_struct else raw_struct)


    if data_type == 'subject':
        raw_style = prop_data.get('style')
    elif data_type == 'comp':
        raw_style = prop_data.get('stories') 
    else: # property_listing
        raw_style = prop_data.get('style', prop_data.get('levels'))
    std['style'] = clean_string(raw_style.split(',')[0] if raw_style and ',' in raw_style else raw_style)


    std['condition'] = clean_string(prop_data.get('condition')) # Will be None for property_listings if not present

    if data_type == 'comp':
        std['sale_price'] = parse_price(prop_data.get('sale_price'))
        std['sale_date'] = clean_string(prop_data.get('sale_date'))
    elif data_type == 'property_listing':
        std['sale_price'] = parse_price(prop_data.get('close_price'))
        std['sale_date'] = clean_string(prop_data.get('close_date'))
    else:
        std['sale_price'] = None
        std['sale_date'] = None


    if data_type == 'property_listing':
        std['latitude'] = prop_data.get('latitude')
        std['longitude'] = prop_data.get('longitude')
    elif data_type == 'subject': # Already handled in process_appraisal_data
        std['latitude'] = subject_lat
        std['longitude'] = subject_lon
    else: # Comps often don't have explicit lat/lon in this dataset
        std['latitude'] = None
        std['longitude'] = None
        
    # Calculate distance for property_listing type if subject lat/lon are available
    if data_type == 'property_listing':
        std['distance_to_subject_km'] = haversine_distance(subject_lat, subject_lon, std.get('latitude'), std.get('longitude'))
    elif data_type == 'comp': # Already provided
        dist_str = prop_data.get('distance_to_subject', '').replace('KM','').strip()
        try:
            std['distance_to_subject_km'] = float(dist_str) if dist_str else None
        except ValueError:
            std['distance_to_subject_km'] = None
    else: # Subject
        std['distance_to_subject_km'] = 0.0


    return std

def process_appraisal_data(appraisal_record: Dict[str, Any], 
                           manual_subject_lat: Optional[float] = None, 
                           manual_subject_lon: Optional[float] = None) -> Dict[str, Any]:
    processed_data = {}
    appraisal_effective_date = appraisal_record.get('subject', {}).get('effective_date')
    reference_year = parse_effective_date_year(appraisal_effective_date)

    subject_data = appraisal_record.get('subject')
    current_subject_lat = manual_subject_lat
    current_subject_lon = manual_subject_lon

    if subject_data:
        # If subject doesn't have lat/lon in its own data, use the manual ones for itself too
        std_subject = standardize_property_features(subject_data, "subject", reference_year, 
                                                    subject_lat=current_subject_lat, 
                                                    subject_lon=current_subject_lon)
        if std_subject:
            processed_data['subject'] = std_subject
            # Ensure current_subject_lat/lon are from the standardized subject if they were found/geocoded
            # For now, we assume manual_subject_lat/lon are the "true" ones for the subject
    
    actual_comps_data = appraisal_record.get('comps', [])
    processed_data['actual_comps'] = []
    for comp_data in actual_comps_data:
        std_comp = standardize_property_features(comp_data, "comp", reference_year)
        if std_comp:
            processed_data['actual_comps'].append(std_comp)

    properties_data = appraisal_record.get('properties', [])
    processed_data['candidate_properties'] = []
    seen_ids_or_addresses = set()

    for prop_listing_data in properties_data:
        std_prop = standardize_property_features(
            prop_listing_data, 
            "property_listing", 
            reference_year,
            subject_lat=current_subject_lat, 
            subject_lon=current_subject_lon
        )
        if std_prop:
            unique_key = std_prop.get('id') or std_prop.get('address_full')
            if unique_key and unique_key in seen_ids_or_addresses:
                continue
            if unique_key:
                seen_ids_or_addresses.add(unique_key)
            processed_data['candidate_properties'].append(std_prop)
            
    return processed_data

# --- k-NN Similarity Finder ---
def find_similar_properties(subject_prop: Dict[str, Any], 
                            candidate_props_list: List[Dict[str, Any]],
                            num_features: List[str], 
                            cat_features: List[str],
                            n_results: int = N_COMPS_TO_FIND) -> List[Dict[str, Any]]:
    if not candidate_props_list:
        return []

    # Create DataFrames
    df_subject = pd.DataFrame([subject_prop])
    df_candidates = pd.DataFrame(candidate_props_list)

    # Ensure all feature columns exist, fill with NaN if not (though standardization should handle this)
    all_feature_names = num_features + cat_features
    for feature in all_feature_names:
        if feature not in df_subject.columns:
            df_subject[feature] = np.nan
        if feature not in df_candidates.columns:
            df_candidates[feature] = np.nan
            
    # Combine subject and candidates for fitting preprocessor to see all categories/ranges
    df_full = pd.concat([df_subject[all_feature_names], df_candidates[all_feature_names]], ignore_index=True)

    # Preprocessing pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier debugging
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ], remainder='drop') # Drop other columns

    # Fit preprocessor on the combined data
    preprocessor.fit(df_full)
    
    # Transform subject and candidate data separately
    subject_transformed = preprocessor.transform(df_subject[all_feature_names])
    candidates_transformed = preprocessor.transform(df_candidates[all_feature_names])

    if candidates_transformed.shape[0] == 0: # No candidates after filtering or empty input
        return []

    # k-NN model
    # Adjust n_neighbors if fewer candidates than n_results
    actual_n_neighbors = min(n_results, candidates_transformed.shape[0])
    if actual_n_neighbors == 0:
        return []
        
    nn_model = NearestNeighbors(n_neighbors=actual_n_neighbors, metric='euclidean')
    nn_model.fit(candidates_transformed)

    distances, indices = nn_model.kneighbors(subject_transformed)
    
    # Return the original dicts of the chosen candidates
    selected_indices = indices[0]
    return [candidate_props_list[i] for i in selected_indices]


# --- Main Execution Example ---
if __name__ == "__main__":
    FILE_PATH = 'appraisals_dataset.json' 

    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {FILE_PATH} not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FILE_PATH}.")
        exit()

    if not data.get('appraisals'):
        print("No appraisals found in the JSON data.")
        exit()

    first_appraisal = data['appraisals'][0]
    
    # Geocoding placeholder for the subject property
    # Real scenario: geocode first_appraisal['subject']['address']
    DEMO_SUBJECT_LAT = 44.2400 
    DEMO_SUBJECT_LON = -76.5600 

    print(f"Processing Appraisal Order ID: {first_appraisal.get('orderID')}")
    print(f"Assuming Subject Lat/Lon for demo: {DEMO_SUBJECT_LAT}, {DEMO_SUBJECT_LON}\\n")

    processed_appraisal = process_appraisal_data(first_appraisal, DEMO_SUBJECT_LAT, DEMO_SUBJECT_LON)

    subject_property = processed_appraisal.get('subject')
    actual_comps = processed_appraisal.get('actual_comps', [])
    candidate_properties = processed_appraisal.get('candidate_properties', [])

    if not subject_property:
        print("Could not process subject property. Exiting.")
        exit()
    
    print("--- Standardized Subject Property (Key Features) ---")
    subject_summary = {k: subject_property.get(k) for k in ['id', 'address_full', 'gla', 'age', 'total_baths', 'structure_type', 'style', 'condition', 'latitude', 'longitude', 'distance_to_subject_km']}
    print(json.dumps(subject_summary, indent=2))
    print("\\n")

    print(f"--- Actual Comps Chosen by Appraiser ({len(actual_comps)}) ---")
    for i, comp in enumerate(actual_comps):
        comp_summary = {k: comp.get(k) for k in ['id', 'address_full', 'gla', 'age', 'total_baths', 'structure_type', 'style', 'condition', 'sale_price', 'distance_to_subject_km']}
        print(f"Actual Comp {i+1}: {json.dumps(comp_summary, indent=2)}")
    print("\\n")
    
    if not candidate_properties:
        print("No candidate properties to search from. Exiting.")
        exit()

    print(f"--- Finding {N_COMPS_TO_FIND} Similar Properties from {len(candidate_properties)} Candidates ---")
    
    # Ensure candidate_properties have the necessary features, even if None (handled by imputer)
    for prop in candidate_properties:
        for feature_key in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
            if feature_key not in prop:
                prop[feature_key] = None # Imputer will handle this

    # The subject property should also have all features defined
    for feature_key in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        if feature_key not in subject_property:
            subject_property[feature_key] = None


    recommended_comps = find_similar_properties(
        subject_prop=subject_property,
        candidate_props_list=candidate_properties,
        num_features=NUMERICAL_FEATURES,
        cat_features=CATEGORICAL_FEATURES,
        n_results=N_COMPS_TO_FIND
    )

    print(f"--- Algorithm Recommended Comps ({len(recommended_comps)}) ---")
    if recommended_comps:
        for i, r_comp in enumerate(recommended_comps):
            r_comp_summary = {k: r_comp.get(k) for k in ['id', 'address_full', 'gla', 'age', 'total_baths', 'structure_type', 'style', 'condition', 'sale_price', 'latitude', 'longitude', 'distance_to_subject_km']}
            print(f"Recommended Comp {i+1}: {json.dumps(r_comp_summary, indent=2)}")
    else:
        print("No comps were recommended by the algorithm.")
    print("\\n")

    print("--- Comparison ---")
    actual_comp_ids = {comp.get('address_full') for comp in actual_comps if comp.get('address_full')} # Use address as ID if available
    recommended_comp_ids = {comp.get('address_full') for comp in recommended_comps if comp.get('address_full')}
    
    matches = actual_comp_ids.intersection(recommended_comp_ids)
    print(f"Number of actual comps: {len(actual_comp_ids)}")
    print(f"Number of recommended comps: {len(recommended_comp_ids)}")
    print(f"Number of matches (by address): {len(matches)}")
    if matches:
        print(f"Matching addresses: {matches}")

    print("\\nData processing and k-NN recommendation complete.")
    print("Further improvements could involve: geocoding subject addresses, refining feature weights, using more advanced similarity metrics (like Gower distance for mixed types without one-hot encoding), or hyperparameter tuning.")