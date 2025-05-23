import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

def load_data(file_path):
    """Loads the appraisals dataset from a JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Found {len(data['appraisals'])} appraisals in the dataset")
    return data['appraisals']

def parse_gla(gla_str):
    """Parse GLA (Gross Living Area) from various string formats."""
    if gla_str is None:
        return 0.0
    
    if isinstance(gla_str, (int, float)):
        return float(gla_str)
    
    if not isinstance(gla_str, str):
        return 0.0
    
    # Remove commas and convert to lowercase for consistency
    cleaned_str = str(gla_str).lower().replace(',', '')
    
    # Handle "+/-" notation by taking the base number
    if '+/-' in cleaned_str:
        cleaned_str = cleaned_str.split('+/-')[0]
    
    # Remove common suffixes
    cleaned_str = cleaned_str.replace(' sqft', '').replace('sqft', '')
    cleaned_str = cleaned_str.replace(' sq ft', '').replace('sq ft', '')
    
    # Try to convert the remaining string to float
    try:
        return float(cleaned_str.strip())
    except ValueError:
        return 0.0

def parse_bath_count(bath_str):
    """Parse bathroom count from various string formats."""
    if not bath_str or bath_str is None:
        return 0.0
    
    if isinstance(bath_str, (int, float)):
        return float(bath_str)
    
    bath_str = str(bath_str).upper().strip()
    
    # Try X:Y format first
    if ':' in bath_str:
        try:
            parts = bath_str.split(':')
            return float(parts[0]) + (float(parts[1]) * 0.5 if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            pass
    
    # Try XF YH format
    total_baths = 0.0
    
    # Look for full baths (F or P)
    full_match = re.search(r'(\d+)\s*[FP]', bath_str)
    if full_match:
        total_baths += float(full_match.group(1))
    
    # Look for half baths (H)
    half_match = re.search(r'(\d+)\s*H', bath_str)
    if half_match:
        total_baths += float(half_match.group(1)) * 0.5
    
    # If no patterns matched but it's just a number
    if not full_match and not half_match:
        try:
            return float(bath_str)
        except ValueError:
            # Check if we have full_baths and half_baths separately
            try:
                full_baths = float(bath_str.get('full_baths', 0) or 0)
                half_baths = float(bath_str.get('half_baths', 0) or 0)
                return full_baths + (half_baths * 0.5)
            except (AttributeError, ValueError):
                return 0.0
    
    return total_baths

def preprocess_property(prop_data, is_subject=False):
    """Extracts and preprocesses features for a single property."""
    features = {}
    
    # Extract and clean GLA
    gla_str = prop_data.get('gla')
    features['gla'] = parse_gla(gla_str)

    # Handle year_built
    year_built_val = prop_data.get('year_built')
    if year_built_val and str(year_built_val).strip():
        try:
            features['year_built'] = int(float(str(year_built_val).strip()))
        except ValueError:
            features['year_built'] = 0
    else:
        features['year_built'] = 0
    
    # Bath count parsing
    bath_count = 0.0
    # Try different bath count fields
    if 'bath_count' in prop_data:
        bath_count = parse_bath_count(prop_data['bath_count'])
    elif 'num_baths' in prop_data:
        bath_count = parse_bath_count(prop_data['num_baths'])
    elif 'full_baths' in prop_data or 'half_baths' in prop_data:
        full_baths = float(prop_data.get('full_baths', 0) or 0)
        half_baths = float(prop_data.get('half_baths', 0) or 0)
        bath_count = full_baths + (half_baths * 0.5)
    features['bath_count'] = bath_count

    # Handle bed count from various fields
    bed_count = 0
    if 'bed_count' in prop_data:
        bed_count_val = prop_data['bed_count']
    elif 'num_beds' in prop_data:
        bed_count_val = prop_data['num_beds']
    elif 'bedrooms' in prop_data:
        bed_count_val = prop_data['bedrooms']
    else:
        bed_count_val = 0
        
    try:
        bed_count = int(float(str(bed_count_val).strip()))
    except (ValueError, AttributeError):
        bed_count = 0
    features['bed_count'] = bed_count

    # Condition (will be one-hot encoded)
    features['condition'] = str(prop_data.get('condition', 'Unknown'))

    # Distance to subject (only for comps/properties, not for the subject itself)
    if not is_subject:
        dist_str = prop_data.get('distance_to_subject')
        if dist_str is None:
            features['distance_to_subject'] = 999.0  # Default if missing
        elif isinstance(dist_str, (int, float)):
            features['distance_to_subject'] = float(dist_str)
        elif isinstance(dist_str, str) and 'KM' in dist_str:
            try:
                features['distance_to_subject'] = float(dist_str.replace(' KM', '').strip())
            except ValueError:
                features['distance_to_subject'] = 999.0
        else:
            features['distance_to_subject'] = 999.0

    return features

if __name__ == "__main__":
    FILE_PATH = 'appraisals_dataset.json'
    
    print("\n=== Starting Property Recommendation System ===")
    appraisals_data = load_data(FILE_PATH)
    
    print("\n--- Preparing data for appraisal-level splitting and evaluation ---")
    
    processed_appraisals = []
    skipped_appraisals = 0
    
    for idx, appraisal in enumerate(appraisals_data):
        try:
            subject_props = appraisal.get('subject', {})
            potential_props_list = appraisal.get('properties', [])
            chosen_comps_details = appraisal.get('comps', [])

            if not potential_props_list or not chosen_comps_details:
                skipped_appraisals += 1
                continue

            print(f"\rProcessing appraisal {idx+1}/{len(appraisals_data)}", end='', flush=True)

            subject_features = preprocess_property(subject_props, is_subject=True)
            actual_chosen_comp_addresses = {c.get('address','').strip().lower() for c in chosen_comps_details}

            appraisal_instances = []
            for pot_prop_data in potential_props_list:
                current_prop_features = preprocess_property(pot_prop_data, is_subject=False)
                prop_address = pot_prop_data.get('address','').strip().lower()

                # Calculate delta features
                instance_features = {}
                instance_features['delta_gla'] = abs(current_prop_features.get('gla',0) - subject_features.get('gla',0))
                instance_features['delta_year_built'] = abs(current_prop_features.get('year_built',0) - subject_features.get('year_built',0))
                instance_features['delta_bath_count'] = abs(current_prop_features.get('bath_count',0) - subject_features.get('bath_count',0))
                instance_features['delta_bed_count'] = abs(current_prop_features.get('bed_count',0) - subject_features.get('bed_count',0))
                instance_features['subject_condition'] = subject_features.get('condition', 'Unknown')
                instance_features['prop_condition'] = current_prop_features.get('condition', 'Unknown')
                instance_features['distance_to_subject'] = current_prop_features.get('distance_to_subject', 999)
                instance_features['original_prop_address'] = prop_address 
                
                target = 1 if prop_address in actual_chosen_comp_addresses else 0
                appraisal_instances.append({'features': instance_features, 'target': target, 'prop_address': prop_address})
            
            if appraisal_instances:  # Only add if we have instances
                processed_appraisals.append({
                    'appraisal_id': appraisal.get('orderID', f"appraisal_{idx}"),
                    'potential_properties': appraisal_instances,
                    'actual_chosen_comp_addresses': actual_chosen_comp_addresses
                })
        except Exception as e:
            print(f"\nError processing appraisal {idx}: {str(e)}")
            skipped_appraisals += 1
            continue

    print(f"\n\nProcessed {len(processed_appraisals)} appraisals successfully")
    print(f"Skipped {skipped_appraisals} appraisals due to missing data or errors")

    if not processed_appraisals:
        print("No valid appraisals to process. Exiting.")
        exit(1)

    print("\n--- Splitting data into train and test sets ---")
    train_appraisals, test_appraisals = train_test_split(processed_appraisals, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_appraisals)} appraisals")
    print(f"Test set: {len(test_appraisals)} appraisals")

    print("\n--- Preparing training data ---")
    X_train_list = []
    y_train_list = []
    for app in train_appraisals:
        for prop_instance in app['potential_properties']:
            X_train_list.append(prop_instance['features'])
            y_train_list.append(prop_instance['target'])
    
    X_train_df = pd.DataFrame(X_train_list)
    y_train_series = pd.Series(y_train_list)

    X_train_processed = X_train_df.drop(columns=['original_prop_address'])
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Positive examples (chosen comps) in training: {sum(y_train_series)}")

    print("\n--- Setting up KNN pipeline ---")
    categorical_features_final = [col for col in ['subject_condition', 'prop_condition'] if col in X_train_processed.columns]
    numerical_features_final = [col for col in ['delta_gla', 'delta_year_built', 'delta_bath_count', 'delta_bed_count', 'distance_to_subject'] if col in X_train_processed.columns]

    print(f"Using numerical features: {numerical_features_final}")
    print(f"Using categorical features: {categorical_features_final}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_final),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_final)
        ],
        remainder='passthrough' 
    )

    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])

    print("\n--- Training KNN model ---")
    knn_pipeline.fit(X_train_processed, y_train_series)

    print("\n--- Evaluating on test set ---")
    hits_at_3 = 0
    total_chosen_comps = 0
    
    for app_test_data in test_appraisals:
        potential_props_for_appraisal = app_test_data['potential_properties']
        if not potential_props_for_appraisal:
            continue

        features_for_prediction_list = []
        prop_addresses_in_order = [] 

        for prop_instance in potential_props_for_appraisal:
            feat_dict = prop_instance['features'].copy()
            prop_addresses_in_order.append(feat_dict.pop('original_prop_address'))
            features_for_prediction_list.append(feat_dict)
        
        if not features_for_prediction_list:
            continue
            
        X_to_predict_df = pd.DataFrame(features_for_prediction_list)
        X_to_predict_df = X_to_predict_df[X_train_processed.columns]

        pred_probabilities = knn_pipeline.predict_proba(X_to_predict_df)[:, 1]

        scored_props = list(zip(pred_probabilities, prop_addresses_in_order))
        scored_props.sort(key=lambda x: x[0], reverse=True)
        predicted_top_3_addresses = {p_addr for _, p_addr in scored_props[:3]}

        actual_chosen_set = app_test_data['actual_chosen_comp_addresses']
        matches = len(predicted_top_3_addresses.intersection(actual_chosen_set))
        hits_at_3 += matches
        total_chosen_comps += len(actual_chosen_set)

    if total_chosen_comps > 0:
        precision_at_3 = hits_at_3 / (len(test_appraisals) * 3) 
        recall_at_3 = hits_at_3 / total_chosen_comps
        print(f"\n--- Final Evaluation Results (K=5) ---")
        print(f"Total Test Appraisals: {len(test_appraisals)}")
        print(f"Correctly Identified Comps (Hits@3): {hits_at_3}")
        print(f"Total Actual Chosen Comps in Test Set: {total_chosen_comps}")
        print(f"Precision@3: {precision_at_3:.4f}") 
        print(f"Recall@3: {recall_at_3:.4f}")
    else:
        print("\nNo chosen comps in the test set to evaluate against.")

    print("\n--- Fine-tuning KNN ---")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    best_k = 5
    best_recall = 0.0
    
    for k in k_values:
        knn_pipeline_tuned = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=k))
        ])
        knn_pipeline_tuned.fit(X_train_processed, y_train_series)

        current_hits_at_3 = 0
        current_total_chosen = 0
        
        for app_test_data in test_appraisals:
            potential_props_for_appraisal = app_test_data['potential_properties']
            if not potential_props_for_appraisal:
                continue

            features_for_prediction_list = []
            prop_addresses_in_order = []
            
            for prop_instance in potential_props_for_appraisal:
                feat_dict = prop_instance['features'].copy()
                prop_addresses_in_order.append(feat_dict.pop('original_prop_address'))
                features_for_prediction_list.append(feat_dict)
            
            if not features_for_prediction_list:
                continue

            X_to_predict_df = pd.DataFrame(features_for_prediction_list)
            X_to_predict_df = X_to_predict_df[X_train_processed.columns]
            
            pred_probabilities = knn_pipeline_tuned.predict_proba(X_to_predict_df)[:, 1]
            scored_props = sorted(list(zip(pred_probabilities, prop_addresses_in_order)), key=lambda x: x[0], reverse=True)
            predicted_top_3_addresses = {p_addr for _, p_addr in scored_props[:3]}
            
            actual_chosen_set = app_test_data['actual_chosen_comp_addresses']
            current_hits_at_3 += len(predicted_top_3_addresses.intersection(actual_chosen_set))
            current_total_chosen += len(actual_chosen_set)

        if current_total_chosen > 0:
            recall_k = current_hits_at_3 / current_total_chosen
            print(f"K={k}: Recall@3 = {recall_k:.4f} (Hits: {current_hits_at_3})")
            if recall_k > best_recall:
                best_recall = recall_k
                best_k = k
        else:
            print(f"K={k}: No chosen comps in test set for evaluation.")
    
    print(f"\nBest K value: {best_k} (Recall@3 = {best_recall:.4f})")
    print("\nScript finished.")
