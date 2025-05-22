import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

from data_loader import AppraisalDataLoader
from knn_recommender import KNNPropertyRecommender
from neural_recommender import NeuralPropertyRecommender
from llm_recommender import LLMPropertyRecommender
from llama_recommender import LlamaPropertyRecommender

# Configure page settings
st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up session state for persistent variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'appraisals' not in st.session_state:
    st.session_state.appraisals = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'location_data' not in st.session_state:
    st.session_state.location_data = None
if 'query_property' not in st.session_state:
    st.session_state.query_property = None
if 'query_features' not in st.session_state:
    st.session_state.query_features = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

# Title and description
st.title("🏠 Property Recommendation System")
st.markdown("""
This application helps you find similar properties (comps) using four different methods:
- **KNN**: K-Nearest Neighbors using direct property features
- **Neural Network**: Deep learning approach using a siamese network
- **LLM**: Language model-based semantic similarity
- **LLAMA**: Fine-tuned LLAMA model using Unsloth for property comparables
""")

# Sidebar for data loading and configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data loading
    data_path = st.text_input("Data Path", value="src/data/appraisals_dataset.json")
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            try:
                # Create data loader
                data_loader = AppraisalDataLoader(data_path=data_path)
                
                # Load appraisal data
                appraisals = data_loader.load_data()
                appraisals_df = data_loader.to_dataframe()
                
                # Preprocess features
                features = data_loader.preprocess_features()
                feature_names = data_loader.get_feature_names()
                
                # Preprocess location data
                location_data = data_loader.preprocess_location_data()
                
                # Store in session state
                st.session_state.appraisals = appraisals
                st.session_state.features = features
                st.session_state.feature_names = feature_names
                st.session_state.location_data = location_data
                st.session_state.data_loaded = True
                
                # Success message
                st.success(f"Successfully loaded {len(appraisals)} properties with {features.shape[1]} features.")
                
                # Add properties to session state for selection
                order_ids = [prop.get('orderID') for prop in appraisals]
                addresses = [prop.get('subject', {}).get('address', 'Unknown') for prop in appraisals]
                property_info = [f"{id} - {addr}" for id, addr in zip(order_ids, addresses)]
                st.session_state.property_info = property_info
                st.session_state.order_ids = order_ids
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Method selection
    st.subheader("Methods")
    knn_enabled = st.checkbox("KNN", value=True)
    neural_enabled = st.checkbox("Neural Network", value=True)
    llm_enabled = st.checkbox("LLM", value=True)
    llama_enabled = st.checkbox("LLAMA", value=False)
    
    # Location filtering
    st.subheader("Location Filtering")
    filter_by_location = st.checkbox("Filter by Location", value=False)
    
    if filter_by_location:
        city = st.text_input("City")
        province = st.text_input("Province (e.g., ON, BC)")
        postal_code = st.text_input("Postal Code (e.g., K7M)")
        municipality = st.text_input("Municipality/District")
    else:
        city, province, postal_code, municipality = "", "", "", ""
    
    # Advanced settings collapsible
    with st.expander("Advanced Settings"):
        # KNN Settings
        st.subheader("KNN Settings")
        knn_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5)
        knn_metric = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"], index=1)
        knn_p = st.slider("p (Power Parameter)", min_value=1, max_value=10, value=2, 
                         help="p=1 for Manhattan, p=2 for Euclidean")
        
        # Neural Network Settings
        st.subheader("Neural Network Settings")
        nn_embedding_dim = st.slider("Embedding Dimension", min_value=16, max_value=256, value=64, step=16)
        nn_hidden_units = st.text_input("Hidden Units (comma-separated)", value="128,64")
        nn_epochs = st.slider("Epochs", min_value=5, max_value=100, value=20)
        nn_early_stopping = st.checkbox("Early Stopping", value=True)
        
        # LLM Settings
        st.subheader("LLM Settings")
        llm_model = st.selectbox("Model", ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0)
        llm_cache = st.checkbox("Cache Embeddings", value=True)
        llm_cache_dir = st.text_input("Cache Directory", value="cache")
        
        # LLAMA Settings
        st.subheader("LLAMA Settings")
        llama_model = st.selectbox("LLAMA Model", ["unsloth/llama-3-8b-bnb-4bit", "unsloth/mistral-7b-bnb-4bit"], index=0)
        llama_cache_dir = st.text_input("LLAMA Cache Directory", value="cache/llama")
        llama_epochs = st.slider("LLAMA Fine-tuning Epochs", min_value=1, max_value=10, value=3)
        llama_batch_size = st.slider("LLAMA Batch Size", min_value=1, max_value=8, value=4)
        llama_learning_rate = st.number_input("LLAMA Learning Rate", min_value=1e-5, max_value=1e-3, value=2e-4, format="%.5f")
        llama_lora_rank = st.slider("LLAMA LoRA Rank", min_value=4, max_value=64, value=16, step=4)
        
        # Feature Weights
        st.subheader("Feature Weights")
        feature_weights_str = st.text_area("Feature Weights (JSON)", value="", 
                                        help='e.g., {"subject_year_built": 1.5, "subject_style": 2.0}')
        
        # Number of results
        st.subheader("Results")
        top_n = st.slider("Number of Results", min_value=1, max_value=20, value=5)
        detailed_explanations = st.checkbox("Detailed Explanations", value=True)

# Main content area
if not st.session_state.data_loaded:
    st.info("Please load the appraisal data using the sidebar.")
else:
    # Property selection tab and property details input tab
    tab1, tab2, tab3 = st.tabs(["Select Existing Property", "Enter Property Details", "Chat with LLM"])
    
    # Tab 1: Select existing property
    with tab1:
        st.header("Select a Property")
        
        # Property selection
        selected_property_idx = st.selectbox(
            "Choose a property:",
            options=range(len(st.session_state.property_info)),
            format_func=lambda x: st.session_state.property_info[x]
        )
        
        # Display selected property details
        if selected_property_idx is not None:
            selected_property = st.session_state.appraisals[selected_property_idx]
            subject = selected_property.get('subject', {})
            
            # Create two columns for property details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Property Details")
                st.write(f"**Address:** {subject.get('address', 'N/A')}")
                st.write(f"**City/Municipality:** {subject.get('municipality_district', 'N/A')}")
                st.write(f"**Structure Type:** {subject.get('structure_type', 'N/A')}")
                st.write(f"**Style:** {subject.get('style', 'N/A')}")
                st.write(f"**Year Built:** {subject.get('year_built', 'N/A')}")
                st.write(f"**Living Area (sq ft):** {subject.get('gla', 'N/A')}")
            
            with col2:
                st.subheader("Additional Information")
                st.write(f"**Lot Size (sq ft):** {subject.get('lot_size_sf', 'N/A')}")
                st.write(f"**Bedrooms:** {subject.get('num_beds', 'N/A')}")
                st.write(f"**Bathrooms:** {subject.get('num_baths', 'N/A')}")
                st.write(f"**Condition:** {subject.get('condition', 'N/A')}")
                st.write(f"**Basement:** {subject.get('basement', 'N/A')}")
                st.write(f"**Construction:** {subject.get('construction', 'N/A')}")
            
            # Set as query property
            if st.button("Set as Query Property"):
                st.session_state.query_property = selected_property
                
                # Get query features
                query_features = st.session_state.features[selected_property_idx]
                st.session_state.query_features = query_features
                
                st.success(f"Property set as query: {subject.get('address', 'N/A')}")
    
    # Tab 2: Enter property details manually
    with tab2:
        st.header("Enter Property Details")
        st.write("Not yet implemented - Future enhancement.")
        
        # This would be a form for manually entering property details
        # For now, we'll just show a placeholder
        st.info("This feature will allow you to input property details to find similar properties even if they're not in the dataset.")
    
    # Tab 3: Chat with LLM for recommendations
    with tab3:
        st.header("Chat with LLM")
        
        # Simple chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I can help you find similar properties. Please describe the property you're looking for, or ask about specific features you're interested in."}
            ]
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat input
        if prompt := st.chat_input("What property are you looking for?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Add assistant response to chat history
            with st.chat_message("assistant"):
                if not llm_enabled:
                    response = "Please enable the LLM method in the sidebar settings to use the chat interface."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    with st.spinner("Processing..."):
                        # Here you would normally process with the LLM
                        # For now, we'll just return a placeholder message
                        response = "I'll help you find properties based on your description. This feature requires integration with a local LLM, which will be added in a future update."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Run recommendations section
    st.header("Generate Recommendations")
    
    if st.session_state.query_property is not None:
        query_property = st.session_state.query_property
        query_subject = query_property.get('subject', {})
        
        st.write(f"Query Property: **{query_subject.get('address', 'N/A')}**")
        
        # Location parameters for filtering
        location_params = {}
        if filter_by_location:
            if city:
                location_params['city'] = city
            if province:
                location_params['province'] = province
            if postal_code:
                location_params['postal_code'] = postal_code
            if municipality:
                location_params['municipality'] = municipality
        
        # Parse feature weights
        feature_weights = {}
        if feature_weights_str:
            try:
                feature_weights = json.loads(feature_weights_str)
            except json.JSONDecodeError:
                st.error("Invalid JSON for feature weights. Please correct the format.")
        
        # Run button
        if st.button("Find Similar Properties"):
            with st.spinner("Generating recommendations..."):
                try:
                    # Clear previous recommendations
                    st.session_state.recommendations = {}
                    
                    # Run KNN if enabled
                    if knn_enabled:
                        # Parse hidden units for neural network (needed for both methods)
                        hidden_units = [int(u) for u in nn_hidden_units.split(',')]
                        
                        # Create and fit the KNN recommender
                        knn_recommender = KNNPropertyRecommender(
                            n_neighbors=knn_neighbors, 
                            metric=knn_metric, 
                            p=knn_p
                        )
                        knn_recommender.fit(
                            st.session_state.features, 
                            st.session_state.feature_names, 
                            feature_weights
                        )
                        
                        # Get recommendations
                        knn_recommendations = knn_recommender.recommend(
                            query_property=query_property,
                            property_data=st.session_state.appraisals,
                            query_features=st.session_state.query_features,
                            filter_by_location=filter_by_location,
                            location_params=location_params
                        )
                        
                        # Limit to top_n
                        knn_recommendations = knn_recommendations[:top_n]
                        
                        # Generate explanations if requested
                        knn_explanations = None
                        if detailed_explanations and knn_recommendations:
                            knn_explanations = knn_recommender.explain_recommendations(
                                query_property=query_property,
                                recommendations=knn_recommendations,
                                top_n_features=5
                            )
                        
                        # Store in session state
                        st.session_state.recommendations['knn'] = {
                            'recommendations': knn_recommendations,
                            'explanations': knn_explanations
                        }
                    
                    # Run Neural Network if enabled
                    if neural_enabled:
                        # Parse hidden units for neural network
                        hidden_units = [int(u) for u in nn_hidden_units.split(',')]
                        
                        # Create and fit the neural recommender
                        neural_recommender = NeuralPropertyRecommender(
                            embedding_dim=nn_embedding_dim, 
                            hidden_units=hidden_units
                        )
                        
                        # Train the model
                        neural_recommender.fit(
                            features=st.session_state.features,
                            feature_names=st.session_state.feature_names,
                            importance_weights=feature_weights,
                            epochs=nn_epochs,
                            early_stopping=nn_early_stopping,
                            verbose=1
                        )
                        
                        # Get recommendations
                        neural_recommendations = neural_recommender.recommend(
                            query_property=query_property,
                            property_data=st.session_state.appraisals,
                            all_features=st.session_state.features,
                            query_features=st.session_state.query_features,
                            filter_by_location=filter_by_location,
                            location_params=location_params,
                            top_n=top_n
                        )
                        
                        # Generate explanations if requested
                        neural_explanations = None
                        if detailed_explanations and neural_recommendations:
                            neural_explanations = neural_recommender.explain_recommendations(
                                query_property=query_property,
                                recommendations=neural_recommendations,
                                top_n_features=5
                            )
                        
                        # Store in session state
                        st.session_state.recommendations['neural'] = {
                            'recommendations': neural_recommendations,
                            'explanations': neural_explanations
                        }
                    
                    # Run LLM if enabled
                    if llm_enabled:
                        # Create cache directory if it doesn't exist and caching is enabled
                        if llm_cache and llm_cache_dir:
                            os.makedirs(llm_cache_dir, exist_ok=True)
                        
                        # Create and fit the LLM recommender
                        llm_recommender = LLMPropertyRecommender(
                            model_name=llm_model,
                            cache_embeddings=llm_cache,
                            cache_dir=llm_cache_dir
                        )
                        
                        # Update feature weights if provided
                        if feature_weights:
                            # Convert feature names from 'subject_X' to just 'X'
                            converted_weights = {}
                            for key, value in feature_weights.items():
                                if key.startswith('subject_'):
                                    converted_key = key[8:]  # Remove 'subject_' prefix
                                    converted_weights[converted_key] = value
                                else:
                                    converted_weights[key] = value
                                    
                            llm_recommender.update_feature_weights(converted_weights)
                        
                        # Fit the model
                        llm_recommender.fit(st.session_state.appraisals)
                        
                        # Get recommendations
                        llm_recommendations = llm_recommender.recommend(
                            query_property=query_property,
                            top_n=top_n,
                            filter_by_location=filter_by_location,
                            location_params=location_params
                        )
                        
                        # Generate explanations if requested
                        llm_explanations = None
                        if detailed_explanations and llm_recommendations:
                            llm_explanations = [
                                llm_recommender.generate_explanation(
                                    query_property=query_property,
                                    recommendation=rec,
                                    detailed=True
                                ) for rec in llm_recommendations
                            ]
                        
                        # Store in session state
                        st.session_state.recommendations['llm'] = {
                            'recommendations': llm_recommendations,
                            'explanations': llm_explanations
                        }
                    
                    # Run LLAMA if enabled
                    if llama_enabled:
                        with st.spinner("Running LLAMA recommender (this may take time for fine-tuning)..."):
                            try:
                                llama_recommender = LlamaPropertyRecommender(
                                    model_name=llama_model,
                                    cache_dir=llama_cache_dir
                                )
                                llama_recommender.fit(
                                    properties=st.session_state.appraisals,
                                    lr=llama_learning_rate,
                                    num_epochs=llama_epochs,
                                    batch_size=llama_batch_size,
                                    lora_rank=llama_lora_rank
                                )
                                llama_recommendations = llama_recommender.recommend(
                                    query_property=query_property,
                                    top_n=top_n,
                                    filter_by_location=filter_by_location,
                                    location_params=location_params
                                )
                                
                                if detailed_explanations:
                                    llama_explanations = []
                                    for rec in llama_recommendations:
                                        exp = llama_recommender.generate_explanation(
                                            query_property=query_property,
                                            recommendation=rec,
                                            detailed=True
                                        )
                                        llama_explanations.append(exp)
                                    st.session_state.recommendations['llama'] = {
                                        'recommendations': llama_recommendations,
                                        'explanations': llama_explanations
                                    }
                                
                                st.success("Recommendations generated successfully!")
                            except Exception as e:
                                st.error(f"Error running LLAMA recommender: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
        
        # Display recommendations
        if st.session_state.recommendations:
            # Create tabs for each method
            method_tabs = []
            
            if 'knn' in st.session_state.recommendations:
                method_tabs.append("KNN")
            if 'neural' in st.session_state.recommendations:
                method_tabs.append("Neural Network")
            if 'llm' in st.session_state.recommendations:
                method_tabs.append("LLM")
            if 'llama' in st.session_state.recommendations:
                method_tabs.append("LLAMA")
                
            if method_tabs:
                tabs = st.tabs(method_tabs)
                
                for i, method in enumerate(method_tabs):
                    with tabs[i]:
                        method_key = method.lower().replace(" ", "_")
                        if method_key == "neural_network":
                            method_key = "neural"
                            
                        if method_key in st.session_state.recommendations:
                            recommendations = st.session_state.recommendations[method_key]['recommendations']
                            explanations = st.session_state.recommendations[method_key]['explanations']
                            
                            # Display recommendations in a table
                            st.subheader(f"{method} Recommendations")
                            
                            # Convert to DataFrame for display
                            rows = []
                            for j, rec in enumerate(recommendations):
                                subject = rec.get('subject', {})
                                row = {
                                    "Rank": j+1,
                                    "Property ID": rec.get('orderID'),
                                    "Address": subject.get('address', 'N/A'),
                                    "Type": subject.get('structure_type', 'N/A'),
                                    "Style": subject.get('style', 'N/A'),
                                    "Year Built": subject.get('year_built', 'N/A'),
                                    "GLA (sq ft)": subject.get('gla', 'N/A'),
                                    "Bedrooms": subject.get('num_beds', 'N/A'),
                                    "Bathrooms": subject.get('num_baths', 'N/A'),
                                    "Similarity": f"{rec.get('similarity_score', 0):.4f}"
                                }
                                rows.append(row)
                                
                            if rows:
                                df = pd.DataFrame(rows)
                                st.dataframe(df)
                                
                                # Display explanations for each recommendation
                                if detailed_explanations and explanations:
                                    st.subheader("Explanations")
                                    for j, rec in enumerate(recommendations[:min(3, len(recommendations))]):
                                        with st.expander(f"Explanation for Property {rec.get('orderID')}"):
                                            if j < len(explanations):
                                                explanation = explanations[j]
                                                
                                                if isinstance(explanation, str):
                                                    # LLM explanations are strings
                                                    st.write(explanation)
                                                elif isinstance(explanation, dict):
                                                    # KNN/Neural explanations are dictionaries
                                                    if 'feature_importance' in explanation:
                                                        # Convert to DataFrame for better display
                                                        feature_imp = explanation['feature_importance']
                                                        feature_names = list(feature_imp.keys())
                                                        feature_values = list(feature_imp.values())
                                                        
                                                        # Sort by importance
                                                        sorted_idx = np.argsort(feature_values)[::-1]
                                                        sorted_names = [feature_names[i] for i in sorted_idx]
                                                        sorted_values = [feature_values[i] for i in sorted_idx]
                                                        
                                                        # Create a bar chart
                                                        fig, ax = plt.subplots(figsize=(10, 5))
                                                        ax.barh(sorted_names[:5], sorted_values[:5])
                                                        ax.set_xlabel('Importance')
                                                        ax.set_title('Feature Importance')
                                                        st.pyplot(fig)
                                                    elif 'similar_features' in explanation:
                                                        # Neural network explanations
                                                        similar_features = explanation['similar_features']
                                                        
                                                        # Display as a table
                                                        rows = []
                                                        for feature, details in similar_features.items():
                                                            rows.append({
                                                                "Feature": feature,
                                                                "Similarity": f"{details['similarity']:.2f}",
                                                                "Query Value": details['query_value'],
                                                                "Recommended Value": details['rec_value']
                                                            })
                                                        
                                                        if rows:
                                                            st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No recommendations available. Please run the recommendations first.")
    else:
        st.info("Please select a query property first.")

def get_top_n_recommendations(query_property, top_n=5):
    """Get top-n recommendations using the specified methods."""
    recommendations = {}
    explanations = {}
    
    # Define location parameters
    location_params = {}
    if filter_by_location:
        if city:
            location_params['city'] = city
        if province:
            location_params['province'] = province
        if postal_code:
            location_params['postal_code'] = postal_code
        if municipality:
            location_params['municipality'] = municipality
    
    # Parse feature weights
    feature_weights = {}
    if feature_weights_str:
        try:
            feature_weights = json.loads(feature_weights_str)
        except json.JSONDecodeError:
            st.warning("Invalid JSON format for feature weights. Using default weights.")
    
    # Get recommendations using KNN
    if knn_enabled:
        with st.spinner("Running KNN recommender..."):
            try:
                knn_recommender = KNNPropertyRecommender(n_neighbors=knn_neighbors, metric=knn_metric, p=knn_p)
                knn_recommender.fit(st.session_state.features, st.session_state.appraisals, st.session_state.feature_names)
                knn_recs = knn_recommender.recommend(
                    query_features=st.session_state.query_features,
                    top_n=top_n,
                    filter_by_location=filter_by_location,
                    location_data=st.session_state.location_data,
                    location_params=location_params,
                    feature_weights=feature_weights
                )
                recommendations['KNN'] = knn_recs
                
                if detailed_explanations:
                    knn_exps = []
                    for rec in knn_recs:
                        exp = knn_recommender.generate_explanation(
                            query_property=query_property,
                            recommendation=rec,
                            feature_importance=True
                        )
                        knn_exps.append(exp)
                    explanations['KNN'] = knn_exps
            except Exception as e:
                st.error(f"Error running KNN recommender: {str(e)}")
    
    # Get recommendations using Neural Network
    if neural_enabled:
        with st.spinner("Running Neural Network recommender..."):
            try:
                hidden_units = [int(x) for x in nn_hidden_units.split(',')]
                neural_recommender = NeuralPropertyRecommender(
                    embedding_dim=nn_embedding_dim,
                    hidden_units=hidden_units
                )
                neural_recommender.fit(
                    features=st.session_state.features,
                    properties=st.session_state.appraisals,
                    feature_names=st.session_state.feature_names,
                    epochs=nn_epochs,
                    early_stopping=nn_early_stopping
                )
                neural_recs = neural_recommender.recommend(
                    query_features=st.session_state.query_features,
                    top_n=top_n,
                    filter_by_location=filter_by_location,
                    location_data=st.session_state.location_data,
                    location_params=location_params
                )
                recommendations['Neural'] = neural_recs
                
                if detailed_explanations:
                    neural_exps = []
                    for rec in neural_recs:
                        exp = neural_recommender.generate_explanation(
                            query_property=query_property,
                            recommendation=rec
                        )
                        neural_exps.append(exp)
                    explanations['Neural'] = neural_exps
            except Exception as e:
                st.error(f"Error running Neural Network recommender: {str(e)}")
    
    # Get recommendations using LLM
    if llm_enabled:
        with st.spinner("Running LLM recommender..."):
            try:
                llm_recommender = LLMPropertyRecommender(
                    model_name=llm_model,
                    cache_embeddings=llm_cache,
                    cache_dir=llm_cache_dir
                )
                llm_recommender.fit(st.session_state.appraisals)
                llm_recs = llm_recommender.recommend(
                    query_property=query_property,
                    top_n=top_n,
                    filter_by_location=filter_by_location,
                    location_params=location_params,
                    feature_weights=feature_weights
                )
                recommendations['LLM'] = llm_recs
                
                if detailed_explanations:
                    llm_exps = []
                    for rec in llm_recs:
                        exp = llm_recommender.generate_explanation(
                            query_property=query_property,
                            recommendation=rec,
                            detailed=True
                        )
                        llm_exps.append(exp)
                    explanations['LLM'] = llm_exps
            except Exception as e:
                st.error(f"Error running LLM recommender: {str(e)}")
    
    # Get recommendations using LLAMA
    if llama_enabled:
        with st.spinner("Running LLAMA recommender (this may take time for fine-tuning)..."):
            try:
                llama_recommender = LlamaPropertyRecommender(
                    model_name=llama_model,
                    cache_dir=llama_cache_dir
                )
                llama_recommender.fit(
                    properties=st.session_state.appraisals,
                    lr=llama_learning_rate,
                    num_epochs=llama_epochs,
                    batch_size=llama_batch_size,
                    lora_rank=llama_lora_rank
                )
                llama_recs = llama_recommender.recommend(
                    query_property=query_property,
                    top_n=top_n,
                    filter_by_location=filter_by_location,
                    location_params=location_params
                )
                recommendations['LLAMA'] = llama_recs
                
                if detailed_explanations:
                    llama_exps = []
                    for rec in llama_recs:
                        exp = llama_recommender.generate_explanation(
                            query_property=query_property,
                            recommendation=rec,
                            detailed=True
                        )
                        llama_exps.append(exp)
                    explanations['LLAMA'] = llama_exps
            except Exception as e:
                st.error(f"Error running LLAMA recommender: {str(e)}")
    
    return recommendations, explanations

if __name__ == "__main__":
    # This will run the Streamlit app when executed directly
    pass 