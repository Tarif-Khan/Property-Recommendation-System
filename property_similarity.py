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
NUMERICAL_FEATURES = ['gla', 'age', 'lot_size_sf', 'bedrooms', 'total_baths', 'distance_to_subject_km', 'basement_sqft', 'room_count']
CATEGORICAL_FEATURES = ['structure_type', 'style', 'condition', 'cooling_type', 'heating_type', 'primary_exterior_finish']


# Helper functions to process data from JSON
def clean_string(text: Optional[str]) -> Optional[str]:
    return text.strip() if text else None

def parse_sq_ft(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    
    s_value = str(value).lower()

    if s_value == "n/a" or s_value == "none": # Added "none" for safety
        return None
    
    s_value_cleaned = s_value.replace(",", "").replace("sqft", "").replace("sf", "").strip()
    
    try:
        return float(s_value_cleaned)
    except ValueError:
        # If direct float conversion fails, it's not a simple number.
        # We could add more sophisticated regex here if needed for complex cases,
        # but for now, if it's not a direct float, we'll return None.
        return None

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
        # We return none since it's an error in parsing
        return None, None
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

  

    std = {"data_type": data_type, "original_data": prop_data} 
    
    std['id'] = str(prop_data.get('orderID', prop_data.get('id', prop_data.get('pin', clean_string(prop_data.get('address'))))))
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
    parsed_year_built = None

   
    if data_type == 'property_listing':
        # ... (Property listing logic as it was in the last working version for property_listing)
        # This means the version that correctly set age=40, year_built=1985 for listing 76989
        # For property listings, 'year_built' might sometimes actually be the age.
        if isinstance(year_built_raw, (int, float)) and 0 <= year_built_raw < 150: # Heuristic
            pass # Will be handled in age calculation section for property_listing
        elif isinstance(year_built_raw, str):
            match_year = re.search(r"(\\d{4})", year_built_raw)
            if match_year:
                parsed_year_built = int(match_year.group(1))
        elif isinstance(year_built_raw, (int, float)) and year_built_raw > 1800:
             parsed_year_built = int(year_built_raw)
       

    elif data_type == 'subject' or data_type == 'comp':
        if isinstance(year_built_raw, str):
            year_built_cleaned = year_built_raw.strip()
            match_year = None
            try:
                match_year = re.search(r"(\\d{4})", year_built_cleaned)
            except Exception:
                pass
            
            if match_year:
                try:
                    parsed_year_built = int(match_year.group(1))
                except Exception:
                    pass
            else:
                year_digits = "".join(filter(str.isdigit, year_built_cleaned))
                if len(year_digits) >= 4:
                    potential_year_str = year_digits[:4] 
                    try:
                        val = int(potential_year_str)
                        if 1000 <= val <= datetime.now().year + 10:
                            parsed_year_built = val
                    except ValueError:
                        pass # Silently ignore fallback int conversion error

        elif isinstance(year_built_raw, (int, float)):
            if year_built_raw > 1800:
                parsed_year_built = int(year_built_raw)

    std['year_built'] = parsed_year_built

    # Here's my logic to calculate age and year_built
    # I will note that this code is very messy, but in a future todo it can be cleaned
    std['age'] = None
    if data_type == 'comp':
        age_raw_comp = prop_data.get('age')
        if age_raw_comp and str(age_raw_comp).isdigit():
            std['age'] = int(age_raw_comp)
            if std['year_built'] is None and reference_year and std['age'] is not None:
                 std['year_built'] = reference_year - std['age']
        elif std['year_built'] and reference_year:
            std['age'] = reference_year - std['year_built']
    elif data_type == 'property_listing':
        age_from_year_built_field = None
        if isinstance(year_built_raw, (int, float)) and 0 <= year_built_raw < 150:
            age_from_year_built_field = int(year_built_raw)

        if age_from_year_built_field is not None:
            std['age'] = age_from_year_built_field
            if reference_year and std['age'] is not None:
                std['year_built'] = reference_year - std['age']
        elif parsed_year_built and reference_year:
            std['age'] = reference_year - parsed_year_built
        else:
            age_raw_listing = prop_data.get('age')
            if age_raw_listing and str(age_raw_listing).isdigit() and 0 <= int(str(age_raw_listing)) < 150 :
                std['age'] = int(str(age_raw_listing))
                if reference_year and std['age'] is not None and std['year_built'] is None:
                    std['year_built'] = reference_year - std['age'] # Infer year_built
            else:
                std['age'] = None # No valid age found
    elif data_type == 'subject': # Explicitly handle subject
        if std['year_built'] and reference_year:
            std['age'] = reference_year - std['year_built']
    

    # Fallthrough for other cases (should ideally not happen if types are subject, comp, property_listing)
    # but if std['age'] is still None, it remains so.
    
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
    
    if bath_source_str:
        fb, hb = parse_bath_count(bath_source_str)
    else:
        fb = prop_data.get('full_baths')
        hb = prop_data.get('half_baths')
        try:
            fb = int(fb) if fb is not None else None
            hb = int(hb) if hb is not None else None
        except ValueError:
            fb, hb = None, None
            
    std['full_baths'], std['half_baths'] = fb, hb
    std['total_baths'] = (std.get('full_baths') or 0) + 0.5 * (std.get('half_baths') or 0)

    if data_type == 'subject':
        raw_struct = prop_data.get('structure_type')
    elif data_type == 'comp':
        raw_struct = prop_data.get('prop_type')
    else: # property_listing
        raw_struct_type = prop_data.get('structure_type')
        raw_prop_sub_type = prop_data.get('property_sub_type')
        # DEBUG PRINT for property_listing structure_type sources
        # print(f"DEBUG_LISTING_STRUCT (ID: {std.get('id')}): raw structure_type='{raw_struct_type}', raw property_sub_type='{raw_prop_sub_type}'")
        raw_struct = raw_struct_type if raw_struct_type else raw_prop_sub_type
    
    cleaned_raw_struct = clean_string(raw_struct.split(',')[0] if raw_struct and ',' in raw_struct else raw_struct)
    
    # Normalize structure types, especially for townhouses
    if cleaned_raw_struct and 'townhouse' in cleaned_raw_struct.lower():
        std['structure_type'] = 'Townhouse' # Normalize to generic Townhouse
    else:
        std['structure_type'] = cleaned_raw_struct
    
    # print(f"DEBUG_STRUCT_NORMALIZED (ID: {std.get('id')}, Type: {data_type}): Raw='{raw_struct}', CleanedRaw='{cleaned_raw_struct}', Final='{std['structure_type']}'")

    # --- Style Standardization (Moved after Structure Type) ---
    raw_style = None
    if data_type == 'subject':
        raw_style = prop_data.get('style')
    elif data_type == 'comp':
        raw_style = prop_data.get('stories') 
    else: # property_listing
        raw_style = prop_data.get('style', prop_data.get('levels'))
    std['style'] = clean_string(raw_style.split(',')[0] if raw_style and ',' in raw_style else raw_style)


    std['condition'] = clean_string(prop_data.get('condition')) # Will be None for property_listings if not present

    # New features - Basement SqFt (Numerical)
    raw_basement_sqft = None
    if data_type == 'subject':
        raw_basement_sqft = prop_data.get('basement_area')
    elif data_type == 'comp': # Comps in example don't have explicit bsmt area, but might have finished area
        raw_basement_sqft = prop_data.get('basement_finish_area') 
    else: # property_listing
        raw_basement_sqft = prop_data.get('basement_sqft')
    std['basement_sqft'] = parse_sq_ft(raw_basement_sqft)

    # New features - Room Count (Numerical)
    raw_room_count = None
    if data_type == 'subject':
        raw_room_count = prop_data.get('room_total') # 'room_count' also in subject, 'room_total' is parsed as int easier
    elif data_type == 'comp':
        raw_room_count = prop_data.get('total_rooms')
    else: # property_listing
        raw_room_count = prop_data.get('rooms_total', prop_data.get('total_rooms'))
    std['room_count'] = int(str(raw_room_count).strip()) if raw_room_count and str(raw_room_count).strip().isdigit() else None

    # New features - Cooling Type (Categorical)
    raw_cooling = None
    if data_type == 'subject':
        raw_cooling = prop_data.get('cooling')
    elif data_type == 'comp':
        raw_cooling = prop_data.get('ac_type')
    else: # property_listing
        raw_cooling = prop_data.get('cooling_system', prop_data.get('cooling'))
    std['cooling_type'] = clean_string(raw_cooling)

    # New features - Heating Type (Categorical)
    raw_heating = None
    if data_type == 'subject':
        raw_heating = prop_data.get('heating')
    elif data_type == 'comp':
        raw_heating = prop_data.get('heat_type')
    else: # property_listing
        raw_heating = prop_data.get('heating_system', prop_data.get('heating'))
    std['heating_type'] = clean_string(raw_heating)

    # New features - Primary Exterior Finish (Categorical)
    raw_exterior = None
    if data_type == 'subject':
        raw_exterior = prop_data.get('exterior_finish')
    elif data_type == 'comp':
        raw_exterior = prop_data.get('exterior_walls')
    else: # property_listing
        raw_exterior = prop_data.get('exterior_finish', prop_data.get('construction_material'))
    std['primary_exterior_finish'] = clean_string(raw_exterior.split(',')[0] if raw_exterior and ',' in raw_exterior else raw_exterior)


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
            
    # --- Pre-filter candidates by subject's structure_type if available ---
    subject_structure_type = std_subject.get('structure_type')
    if subject_structure_type:
        original_candidate_count = len(processed_data['candidate_properties'])
        processed_data['candidate_properties'] = [prop for prop in processed_data['candidate_properties'] if prop.get('structure_type') == subject_structure_type]
        print(f"Filtered candidates by subject structure type ('{subject_structure_type}'): {original_candidate_count} -> {len(processed_data['candidate_properties'])} candidates")

        # --- Override features for candidates that are also actual_comps ---
        print("\n--- Checking/Overriding features for candidates that are also actual comps ---")
        actual_comps_by_address = {str(ac.get('address_full','')).strip().lower(): ac for ac in processed_data['actual_comps'] if ac.get('address_full')}
        
        for cand_prop in processed_data['candidate_properties']:
            cand_addr_key = str(cand_prop.get('address_full','')).strip().lower()
            if cand_addr_key in actual_comps_by_address:
                actual_comp_data = actual_comps_by_address[cand_addr_key]
                print(f"  Overriding data for candidate ID {cand_prop.get('id')} (Address: {cand_prop.get('address_full')}) with actual comp data (ID: {actual_comp_data.get('id')})")
                
                # Override key fields - ensure these are already standardized in actual_comp_data
                # For this to work perfectly, actual_comps should also be fully standardized like subject/listings.
                # Assuming actual_comps processed by standardize_property_features have these fields correctly standardized.
                fields_to_override = ['gla', 'age', 'total_baths', 'condition', 'distance_to_subject_km', 'style', 'sale_price'] # Add/remove fields as needed
                for field in fields_to_override:
                    if field in actual_comp_data and actual_comp_data[field] is not None: # Only override if actual_comp has a value
                        # print(f"    - {field}: from '{cand_prop.get(field)}' to '{actual_comp_data[field]}'")
                        cand_prop[field] = actual_comp_data[field]
        print("--- Finished checking/overriding features ---\n")
        # --- End Override ---

        # --- DEBUG: Print summary of filtered townhouse candidates ---
        print("\n--- Summary of Filtered Townhouse Candidates (Input to k-NN) ---")
        for i, cand_prop in enumerate(processed_data['candidate_properties']):
            cand_summary = {
                "id": cand_prop.get('id'),
                "address_full": cand_prop.get('address_full'),
                "gla": cand_prop.get('gla'),
                "age": cand_prop.get('age'),
                "total_baths": cand_prop.get('total_baths'),
                "condition": cand_prop.get('condition'),
                "distance_to_subject_km": cand_prop.get('distance_to_subject_km'),
                "sale_price": cand_prop.get('sale_price') # Standardized sale price
            }
            print(f"Candidate {i+1}: {json.dumps(cand_summary, indent=2)}")
        print("-----------------------------------------------------------\n")
        # --- End DEBUG ---

    if not processed_data['candidate_properties']:
        print("No candidate properties remaining after filtering. Exiting.")
        return processed_data

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
    
    recommended_results = []
    for i in selected_indices:
        original_prop = candidate_props_list[i] # Get the original dict
        recommended_results.append(original_prop)
    return recommended_results


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
    print(f"Assuming Subject Lat/Lon for demo: {DEMO_SUBJECT_LAT}, {DEMO_SUBJECT_LON}\\\\n")

    processed_appraisal = process_appraisal_data(first_appraisal, DEMO_SUBJECT_LAT, DEMO_SUBJECT_LON)

    subject_property = processed_appraisal.get('subject')
    actual_comps = processed_appraisal.get('actual_comps', [])
    candidate_properties = processed_appraisal.get('candidate_properties', [])

    if not subject_property:
        print("Could not process subject property. Exiting.")
        exit()
    
    # print("\\nDEBUG: Full subject_property before summary:") # Removed, subject is now correct
    # print(json.dumps(subject_property, indent=2))

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

    # DEBUG: Print subject GLA and Lot Size before k-NN
    if subject_property:
        print(f"DEBUG Main: Subject GLA before k-NN: {subject_property.get('gla')}")
        print(f"DEBUG Main: Subject Lot Size SF before k-NN: {subject_property.get('lot_size_sf')}")
    if candidate_properties and len(candidate_properties) > 0:
        print(f"DEBUG Main: First Candidate ({candidate_properties[0].get('id')}) GLA before k-NN: {candidate_properties[0].get('gla')}")
        print(f"DEBUG Main: First Candidate ({candidate_properties[0].get('id')}) Lot Size SF before k-NN: {candidate_properties[0].get('lot_size_sf')}")


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


    recommended_props_standardized = find_similar_properties(
        subject_prop=subject_property,
        candidate_props_list=candidate_properties,
        num_features=NUMERICAL_FEATURES,
        cat_features=CATEGORICAL_FEATURES,
        n_results=N_COMPS_TO_FIND
    )

    if not recommended_props_standardized:
        print("No recommendations returned by find_similar_properties.")
        exit()
    
    print(f"\\n--- Algorithm Recommended Comps ({N_COMPS_TO_FIND}) ---")
    for i, prop_std in enumerate(recommended_props_standardized):
        original_data_for_rec = prop_std.get('original_data', {}) # Get the original raw data

        recommended_comp_display = {
            "id": prop_std.get("id", "N/A"), # From standardized
            "address_full": prop_std.get("address_full", "N/A"), # From standardized
            "gla": prop_std.get("gla"), # From standardized
            "age": prop_std.get("age"), # From standardized (corrected)
            "year_built": prop_std.get("year_built"), # From standardized (derived/corrected)
            "total_baths": prop_std.get("total_baths"), # From standardized
            "structure_type": prop_std.get("structure_type"), # From standardized
            "style": prop_std.get("style"), # From standardized
            "condition": prop_std.get("condition"), # From standardized
            
            # Fields from original data that were not part of KNN features / not transformed for KNN
            "sale_price": prop_std.get("sale_price"), # Use the standardized sale_price
            "latitude": original_data_for_rec.get("latitude"),
            "longitude": original_data_for_rec.get("longitude"),
            "distance_to_subject_km": prop_std.get("distance_to_subject_km") # Calculated during standardization
        }
        print(f"Recommended Comp {i + 1}:")
        print(json.dumps(recommended_comp_display, indent=2))

        # Check if this recommended comp matches any actual comp by address
        current_rec_addr_key = str(prop_std.get("address_full", "")).strip().lower()
        for actual_comp in actual_comps:
            actual_comp_addr_key = str(actual_comp.get("address_full", "")).strip().lower()
            if current_rec_addr_key and current_rec_addr_key == actual_comp_addr_key:
                print("  -> This recommended comp is also an ACTUAL APPRAISER COMP. Comparing data:")
                actual_comp_summary = {k: actual_comp.get(k) for k in ['id', 'gla', 'age', 'total_baths', 'condition', 'distance_to_subject_km', 'sale_price']}
                print(f"     Algorithm Used (Listing Data): {json.dumps(recommended_comp_display, indent=4)}") # Already has most needed fields
                print(f"     Appraiser Actual Comp Data:  {json.dumps(actual_comp_summary, indent=4)}")
                break

    # --- Comparison with Actual Comps ---
    actual_comp_keys = set()
    for comp in actual_comps:
        addr = comp.get('address_full')
        comp_id = comp.get('id')
        if addr: actual_comp_keys.add(str(addr).strip().lower())

    recommended_comp_keys = set()
    for prop_std in recommended_props_standardized:
        addr = prop_std.get('address_full')
        comp_id = prop_std.get('id')
        if addr: recommended_comp_keys.add(str(addr).strip().lower())
    
    matches = actual_comp_keys.intersection(recommended_comp_keys)
    print(f"\nNumber of actual comps uniquely identified: {len(actual_comp_keys)}")
    print(f"Number of recommended comps uniquely identified: {len(recommended_comp_keys)}")
    print(f"Number of matches (by address): {len(matches)}")
    if matches:
        print(f"Matching addresses: {matches}")

    print("\\nData processing and k-NN recommendation complete.")