import argparse
import json
import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from data_loader import AppraisalDataLoader
from knn_recommender import KNNPropertyRecommender
from neural_recommender import NeuralPropertyRecommender
from llm_recommender import LLMPropertyRecommender
from llama_recommender import LlamaPropertyRecommender


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Property Recommendation System')
    
    # General arguments
    parser.add_argument('--data_path', type=str, default='../data/appraisals_dataset.json',
                        help='Path to the appraisal dataset')
    parser.add_argument('--method', type=str, choices=['knn', 'neural', 'llm', 'llama', 'all'],
                        default='all', help='Recommendation method to use')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    
    # Location filtering
    parser.add_argument('--filter_by_location', action='store_true',
                        help='Filter results by location')
    parser.add_argument('--city', type=str, default='',
                        help='City to filter by')
    parser.add_argument('--province', type=str, default='',
                        help='Province to filter by (e.g., ON, BC)')
    parser.add_argument('--postal_code', type=str, default='',
                        help='Postal code to filter by (e.g., K7M)')
    parser.add_argument('--municipality', type=str, default='',
                        help='Municipality/district to filter by')
    
    # KNN arguments
    parser.add_argument('--knn_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--knn_metric', type=str, default='minkowski',
                        help='Distance metric for KNN')
    parser.add_argument('--knn_p', type=int, default=2,
                        help='Power parameter for Minkowski metric (p=1 for Manhattan, p=2 for Euclidean)')
    
    # Neural network arguments
    parser.add_argument('--nn_embedding_dim', type=int, default=64,
                        help='Embedding dimension for neural network')
    parser.add_argument('--nn_hidden_units', type=str, default='128,64',
                        help='Comma-separated list of hidden units')
    parser.add_argument('--nn_epochs', type=int, default=20,
                        help='Number of epochs for neural network training')
    parser.add_argument('--nn_early_stopping', action='store_true',
                        help='Use early stopping for neural network training')
    
    # LLM arguments
    parser.add_argument('--llm_model', type=str, default='paraphrase-MiniLM-L6-v2',
                        help='Sentence transformer model for LLM-based recommendations')
    parser.add_argument('--llm_cache', action='store_true',
                        help='Cache LLM embeddings')
    parser.add_argument('--llm_cache_dir', type=str, default='../cache',
                        help='Directory to cache LLM embeddings')
    
    # LLAMA arguments
    parser.add_argument('--llama_model', type=str, default='unsloth/llama-3-8b-bnb-4bit',
                        help='LLAMA model for fine-tuning')
    parser.add_argument('--llama_cache_dir', type=str, default='../cache/llama',
                        help='Directory to cache LLAMA model and embeddings')
    parser.add_argument('--llama_epochs', type=int, default=3,
                        help='Number of epochs for LLAMA fine-tuning')
    parser.add_argument('--llama_batch_size', type=int, default=4,
                        help='Batch size for LLAMA fine-tuning')
    parser.add_argument('--llama_learning_rate', type=float, default=2e-4,
                        help='Learning rate for LLAMA fine-tuning')
    parser.add_argument('--llama_lora_rank', type=int, default=16,
                        help='LoRA rank for LLAMA fine-tuning')
    parser.add_argument('--llama_max_length', type=int, default=1024,
                        help='Maximum sequence length for LLAMA model')
    
    # Feature importance
    parser.add_argument('--feature_weights', type=str, default='',
                        help='JSON string of feature weights, e.g., \'{"subject_year_built": 1.5}\'')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate recommendation performance')
    
    # Example query
    parser.add_argument('--query_id', type=str, default=None,
                        help='ID of property to use as query')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of recommendations to return')
    
    # Detailed explanations
    parser.add_argument('--detailed_explanations', action='store_true',
                        help='Generate detailed explanations for recommendations')
    
    return parser.parse_args()


def load_data(data_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str], Dict[str, Any]]:
    """
    Load and preprocess the appraisal data.
    
    Args:
        data_path: Path to the appraisal dataset
        
    Returns:
        Tuple of (appraisals, features, feature_names, location_data)
    """
    logger.info(f"Loading appraisal data from {data_path}")
    
    # Create data loader
    data_loader = AppraisalDataLoader(data_path=data_path)
    
    # Load appraisal data
    appraisals = data_loader.load_data()
    logger.info(f"Loaded {len(appraisals)} appraisal records")
    
    # Convert to DataFrame for easier processing
    appraisals_df = data_loader.to_dataframe()
    logger.info(f"DataFrame shape: {appraisals_df.shape}")
    
    # Preprocess features
    features = data_loader.preprocess_features()
    feature_names = data_loader.get_feature_names()
    logger.info(f"Extracted {features.shape[1]} features")
    
    # Preprocess location data
    location_data = data_loader.preprocess_location_data()
    
    return appraisals, features, feature_names, location_data


def get_query_property(appraisals: List[Dict[str, Any]], query_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a query property by ID or select a random one.
    
    Args:
        appraisals: List of appraisal records
        query_id: ID of property to use as query (optional)
        
    Returns:
        Query property dictionary
    """
    if query_id:
        # Find property with matching ID
        for prop in appraisals:
            if prop.get('orderID') == query_id:
                return prop
        logger.warning(f"Property with ID {query_id} not found. Using a random property instead.")
    
    # Select a random property
    import random
    return random.choice(appraisals)


def get_query_features(query_property: Dict[str, Any], appraisals: List[Dict[str, Any]], 
                      features: np.ndarray) -> np.ndarray:
    """
    Get features for a query property.
    
    Args:
        query_property: Query property dictionary
        appraisals: List of all appraisal records
        features: Feature matrix for all properties
        
    Returns:
        Feature vector for the query property
    """
    # Find the index of the query property in the appraisals list
    query_idx = None
    query_id = query_property.get('orderID')
    
    for i, prop in enumerate(appraisals):
        if prop.get('orderID') == query_id:
            query_idx = i
            break
    
    if query_idx is not None:
        return features[query_idx]
    else:
        # If the query property is not in the appraisals list (e.g., new property),
        # we would need to extract its features using the same preprocessing.
        # For simplicity, we're using a random property's features here.
        logger.warning("Query property not found in dataset. Using random features.")
        return features[0]


def parse_feature_weights(weights_str: str) -> Dict[str, float]:
    """
    Parse feature weights from a JSON string.
    
    Args:
        weights_str: JSON string of feature weights
        
    Returns:
        Dictionary of feature weights
    """
    if not weights_str:
        return {}
        
    try:
        return json.loads(weights_str)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string for feature weights: {weights_str}")
        return {}


def get_location_params(args) -> Dict[str, str]:
    """
    Get location parameters from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of location parameters
    """
    location_params = {}
    
    if args.city:
        location_params['city'] = args.city
        
    if args.province:
        location_params['province'] = args.province
        
    if args.postal_code:
        location_params['postal_code'] = args.postal_code
        
    if args.municipality:
        location_params['municipality'] = args.municipality
        
    return location_params


def run_knn_recommender(appraisals: List[Dict[str, Any]], 
                       features: np.ndarray, 
                       feature_names: List[str], 
                       query_property: Dict[str, Any], 
                       query_features: np.ndarray, 
                       n_neighbors: int, 
                       metric: str,
                       p: int = 2,
                       filter_by_location: bool = False,
                       location_params: Optional[Dict[str, str]] = None,
                       feature_weights: Optional[Dict[str, float]] = None,
                       top_n: int = 5,
                       detailed_explanations: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Dict[str, float]]]]]:
    """
    Run the KNN-based property recommender.
    
    Args:
        appraisals: List of appraisal records
        features: Feature matrix for all properties
        feature_names: List of feature names
        query_property: Query property dictionary
        query_features: Feature vector for the query property
        n_neighbors: Number of neighbors to retrieve
        metric: Distance metric to use
        p: Power parameter for Minkowski metric
        filter_by_location: Whether to filter results by location
        location_params: Location parameters for filtering
        feature_weights: Dictionary of feature weights
        top_n: Number of recommendations to return
        detailed_explanations: Whether to generate detailed explanations
        
    Returns:
        Tuple of (recommendations, explanations)
    """
    logger.info("Running KNN-based recommender...")
    
    # Create and fit the KNN recommender
    knn_recommender = KNNPropertyRecommender(n_neighbors=n_neighbors, metric=metric, p=p)
    knn_recommender.fit(features, feature_names, feature_weights)
    
    # Get recommendations
    recommendations = knn_recommender.recommend(
        query_property=query_property,
        property_data=appraisals,
        query_features=query_features,
        filter_by_location=filter_by_location,
        location_params=location_params
    )
    
    # Limit to top_n
    recommendations = recommendations[:top_n]
    
    # Generate explanations if requested
    explanations = None
    if detailed_explanations and recommendations:
        explanations = knn_recommender.explain_recommendations(
            query_property=query_property,
            recommendations=recommendations,
            top_n_features=5
        )
    
    return recommendations, explanations


def run_neural_recommender(appraisals: List[Dict[str, Any]], 
                          features: np.ndarray, 
                          feature_names: List[str], 
                          query_property: Dict[str, Any], 
                          query_features: np.ndarray, 
                          embedding_dim: int, 
                          hidden_units: List[int], 
                          epochs: int,
                          early_stopping: bool = True,
                          filter_by_location: bool = False,
                          location_params: Optional[Dict[str, str]] = None,
                          feature_weights: Optional[Dict[str, float]] = None,
                          top_n: int = 5,
                          detailed_explanations: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Dict[str, Any]]]]]:
    """
    Run the neural network-based property recommender.
    
    Args:
        appraisals: List of appraisal records
        features: Feature matrix for all properties
        feature_names: List of feature names
        query_property: Query property dictionary
        query_features: Feature vector for the query property
        embedding_dim: Dimension of the property embedding
        hidden_units: List of hidden units for each dense layer
        epochs: Number of epochs for training
        early_stopping: Whether to use early stopping
        filter_by_location: Whether to filter results by location
        location_params: Location parameters for filtering
        feature_weights: Dictionary of feature weights
        top_n: Number of recommendations to return
        detailed_explanations: Whether to generate detailed explanations
        
    Returns:
        Tuple of (recommendations, explanations)
    """
    logger.info("Running neural network-based recommender...")
    
    # Create and fit the neural recommender
    neural_recommender = NeuralPropertyRecommender(
        embedding_dim=embedding_dim, 
        hidden_units=hidden_units
    )
    
    # Train the model
    neural_recommender.fit(
        features=features,
        feature_names=feature_names,
        importance_weights=feature_weights,
        epochs=epochs,
        early_stopping=early_stopping,
        verbose=1
    )
    
    # Get recommendations
    recommendations = neural_recommender.recommend(
        query_property=query_property,
        property_data=appraisals,
        all_features=features,
        query_features=query_features,
        filter_by_location=filter_by_location,
        location_params=location_params,
        top_n=top_n
    )
    
    # Generate explanations if requested
    explanations = None
    if detailed_explanations and recommendations:
        explanations = neural_recommender.explain_recommendations(
            query_property=query_property,
            recommendations=recommendations,
            top_n_features=5
        )
    
    return recommendations, explanations


def run_llm_recommender(appraisals: List[Dict[str, Any]], 
                       query_property: Dict[str, Any],
                       model_name: str, 
                       cache_embeddings: bool,
                       cache_dir: Optional[str] = None,
                       filter_by_location: bool = False,
                       location_params: Optional[Dict[str, str]] = None,
                       feature_weights: Optional[Dict[str, float]] = None,
                       top_n: int = 5,
                       detailed_explanations: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
    """
    Run the LLM-based property recommender.
    
    Args:
        appraisals: List of appraisal records
        query_property: Query property dictionary
        model_name: Name of the sentence transformer model
        cache_embeddings: Whether to cache embeddings
        cache_dir: Directory to cache embeddings
        filter_by_location: Whether to filter results by location
        location_params: Location parameters for filtering
        feature_weights: Dictionary of feature weights
        top_n: Number of recommendations to return
        detailed_explanations: Whether to generate detailed explanations
        
    Returns:
        Tuple of (recommendations, explanations)
    """
    logger.info("Running LLM-based recommender...")
    
    # Create cache directory if it doesn't exist and caching is enabled
    if cache_embeddings and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Create and fit the LLM recommender
    llm_recommender = LLMPropertyRecommender(
        model_name=model_name,
        cache_embeddings=cache_embeddings,
        cache_dir=cache_dir
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
    llm_recommender.fit(appraisals)
    
    # Get recommendations
    recommendations = llm_recommender.recommend(
        query_property=query_property,
        top_n=top_n,
        filter_by_location=filter_by_location,
        location_params=location_params
    )
    
    # Generate explanations if requested
    explanations = None
    if detailed_explanations and recommendations:
        explanations = [
            llm_recommender.generate_explanation(
                query_property=query_property,
                recommendation=rec,
                detailed=True
            ) for rec in recommendations
        ]
    
    return recommendations, explanations


def run_llama_recommender(appraisals: List[Dict[str, Any]], 
                      query_property: Dict[str, Any],
                      model_name: str,
                      cache_dir: Optional[str] = None,
                      max_length: int = 1024,
                      num_epochs: int = 3,
                      batch_size: int = 4,
                      learning_rate: float = 2e-4,
                      lora_rank: int = 16,
                      filter_by_location: bool = False,
                      location_params: Optional[Dict[str, str]] = None,
                      top_n: int = 5,
                      detailed_explanations: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
    """
    Run the LLAMA-based recommender.
    
    Args:
        appraisals: List of appraisal records
        query_property: Query property dictionary
        model_name: Name of the LLAMA model to use
        cache_dir: Directory to cache model and embeddings
        max_length: Maximum token length for input sequences
        num_epochs: Number of epochs for fine-tuning
        batch_size: Batch size for training
        learning_rate: Learning rate for fine-tuning
        lora_rank: Rank for LoRA adapters
        filter_by_location: Whether to filter by location
        location_params: Location parameters for filtering
        top_n: Number of recommendations to return
        detailed_explanations: Whether to generate detailed explanations
        
    Returns:
        Tuple of (recommendations, explanations)
    """
    logger.info("Running LLAMA-based recommender")
    
    # Create recommender
    recommender = LlamaPropertyRecommender(
        model_name=model_name,
        cache_dir=cache_dir,
        max_length=max_length
    )
    
    # Fine-tune on appraisal data
    logger.info("Fine-tuning LLAMA model on appraisal data")
    recommender.fit(
        properties=appraisals,
        lr=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_rank=lora_rank
    )
    
    # Get recommendations
    logger.info("Getting recommendations")
    recommendations = recommender.recommend(
        query_property=query_property,
        top_n=top_n,
        filter_by_location=filter_by_location,
        location_params=location_params
    )
    
    # Generate explanations if requested
    explanations = None
    if recommendations and detailed_explanations:
        logger.info("Generating explanations")
        explanations = []
        for rec in recommendations:
            explanation = recommender.generate_explanation(
                query_property=query_property,
                recommendation=rec,
                detailed=True
            )
            explanations.append(explanation)
    
    return recommendations, explanations


def print_recommendations(recommendations: List[Dict[str, Any]], 
                         method: str, 
                         explanations: Optional[Any] = None) -> None:
    """
    Print recommendations in a human-readable format.
    
    Args:
        recommendations: List of recommended properties
        method: Recommendation method used
        explanations: Explanations for recommendations (if available)
    """
    print(f"\n===== {method.upper()} Recommendations =====")
    
    for i, rec in enumerate(recommendations[:5], 1):
        subject = rec.get('subject', {})
        
        print(f"\n{i}. Property ID: {rec.get('orderID')}")
        print(f"   Address: {subject.get('address', 'N/A')}")
        print(f"   Type: {subject.get('structure_type', 'N/A')}")
        print(f"   Style: {subject.get('style', 'N/A')}")
        print(f"   Year built: {subject.get('year_built', 'N/A')}")
        print(f"   Construction: {subject.get('construction', 'N/A')}")
        print(f"   Similarity score: {rec.get('similarity_score', 0):.4f}")
        
        # Print explanation if available
        if explanations and i-1 < len(explanations):
            if isinstance(explanations[i-1], str):
                # LLM explanations are strings
                print(f"\n   Explanation:\n   {explanations[i-1].replace(chr(10), chr(10)+'   ')}")
            elif isinstance(explanations[i-1], dict):
                # KNN explanations are dictionaries
                print("\n   Top contributing features:")
                if 'feature_importance' in explanations[i-1]:
                    for feature, importance in explanations[i-1]['feature_importance'].items():
                        print(f"   - {feature}: {importance:.2f}")
    
    print("\n")


def save_recommendations(recommendations: List[Dict[str, Any]], 
                        method: str, 
                        output_dir: str, 
                        query_id: str,
                        explanations: Optional[Any] = None) -> None:
    """
    Save recommendations to a JSON file.
    
    Args:
        recommendations: List of recommended properties
        method: Recommendation method used
        output_dir: Directory to save results
        query_id: ID of the query property
        explanations: Explanations for recommendations (if available)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a simplified version of the recommendations for saving
    simplified_recs = []
    for i, rec in enumerate(recommendations):
        subject = rec.get('subject', {})
        
        simplified_rec = {
            'orderID': rec.get('orderID'),
            'address': subject.get('address'),
            'structure_type': subject.get('structure_type'),
            'style': subject.get('style'),
            'year_built': subject.get('year_built'),
            'gla': subject.get('gla'),
            'condition': subject.get('condition'),
            'similarity_score': rec.get('similarity_score')
        }
        
        # Add explanation if available
        if explanations and i < len(explanations):
            if isinstance(explanations[i], str):
                simplified_rec['explanation'] = explanations[i]
            elif isinstance(explanations[i], dict):
                simplified_rec['explanation'] = explanations[i]
        
        simplified_recs.append(simplified_rec)
    
    # Save to file
    filename = f"{output_path}/{method}_recommendations_{query_id}.json"
    with open(filename, 'w') as f:
        json.dump(simplified_recs, f, indent=2)
    
    logger.info(f"Saved recommendations to {filename}")


def main():
    """Main function to run the property recommendation system."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    appraisals, features, feature_names, location_data = load_data(args.data_path)
    
    # Parse feature weights
    feature_weights = parse_feature_weights(args.feature_weights)
    
    # Get location parameters
    location_params = get_location_params(args)
    
    # Get query property
    query_property = get_query_property(appraisals, args.query_id)
    query_id = query_property.get('orderID')
    logger.info(f"Using property {query_id} as query")
    
    # Get query features
    query_features = get_query_features(query_property, appraisals, features)
    
    # Run recommenders based on method
    if args.method in ['knn', 'all']:
        # Parse hidden units for neural network
        hidden_units = [int(u) for u in args.nn_hidden_units.split(',')]
        
        # Run KNN recommender
        knn_recommendations, knn_explanations = run_knn_recommender(
            appraisals=appraisals,
            features=features,
            feature_names=feature_names,
            query_property=query_property,
            query_features=query_features,
            n_neighbors=args.knn_neighbors,
            metric=args.knn_metric,
            p=args.knn_p,
            filter_by_location=args.filter_by_location,
            location_params=location_params,
            feature_weights=feature_weights,
            top_n=args.top_n,
            detailed_explanations=args.detailed_explanations
        )
        
        # Print and save recommendations
        print_recommendations(knn_recommendations, 'knn', knn_explanations)
        save_recommendations(knn_recommendations, 'knn', args.output_dir, query_id, knn_explanations)
    
    if args.method in ['neural', 'all']:
        # Parse hidden units for neural network
        hidden_units = [int(u) for u in args.nn_hidden_units.split(',')]
        
        # Run neural network recommender
        neural_recommendations, neural_explanations = run_neural_recommender(
            appraisals=appraisals,
            features=features,
            feature_names=feature_names,
            query_property=query_property,
            query_features=query_features,
            embedding_dim=args.nn_embedding_dim,
            hidden_units=hidden_units,
            epochs=args.nn_epochs,
            early_stopping=args.nn_early_stopping,
            filter_by_location=args.filter_by_location,
            location_params=location_params,
            feature_weights=feature_weights,
            top_n=args.top_n,
            detailed_explanations=args.detailed_explanations
        )
        
        # Print and save recommendations
        print_recommendations(neural_recommendations, 'neural', neural_explanations)
        save_recommendations(neural_recommendations, 'neural', args.output_dir, query_id, neural_explanations)
    
    if args.method in ['llm', 'all']:
        # Run LLM recommender
        llm_recommendations, llm_explanations = run_llm_recommender(
            appraisals=appraisals,
            query_property=query_property,
            model_name=args.llm_model,
            cache_embeddings=args.llm_cache,
            cache_dir=args.llm_cache_dir,
            filter_by_location=args.filter_by_location,
            location_params=location_params,
            feature_weights=feature_weights,
            top_n=args.top_n,
            detailed_explanations=args.detailed_explanations
        )
        
        # Print and save recommendations
        print_recommendations(llm_recommendations, 'llm', llm_explanations)
        save_recommendations(llm_recommendations, 'llm', args.output_dir, query_id, llm_explanations)
    
    if args.method in ['llama', 'all']:
        # Run LLAMA recommender
        llama_recommendations, llama_explanations = run_llama_recommender(
            appraisals=appraisals,
            query_property=query_property,
            model_name=args.llama_model,
            cache_dir=args.llama_cache_dir,
            max_length=args.llama_max_length,
            num_epochs=args.llama_epochs,
            batch_size=args.llama_batch_size,
            learning_rate=args.llama_learning_rate,
            lora_rank=args.llama_lora_rank,
            filter_by_location=args.filter_by_location,
            location_params=location_params,
            top_n=args.top_n,
            detailed_explanations=args.detailed_explanations
        )
        
        # Print and save recommendations
        print_recommendations(llama_recommendations, 'LLAMA', llama_explanations)
        save_recommendations(llama_recommendations, 'LLAMA', args.output_dir, query_id, llama_explanations)


if __name__ == '__main__':
    main() 