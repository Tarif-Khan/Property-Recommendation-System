import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional

class KNNPropertyRecommender:
    """
    K-Nearest Neighbors based property recommendation system.
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int = 2, **kwargs):
        """
        Initialize the KNN property recommender.
        
        Args:
            n_neighbors: Number of neighbors to retrieve
            metric: Distance metric to use (e.g., 'minkowski', 'euclidean', 'manhattan')
            p: Power parameter for Minkowski metric (p=1 for Manhattan, p=2 for Euclidean)
            **kwargs: Additional parameters for sklearn NearestNeighbors
        """
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, p=p, **kwargs)
        self.scaler = StandardScaler()
        self.is_fit = False
        self.feature_names = None
        self.importance_weights = None
        
    def fit(self, 
            features: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            importance_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Fit the KNN model to the property feature data.
        
        Args:
            features: Feature matrix for properties (n_samples x n_features)
            feature_names: Names of the features
            importance_weights: Dictionary mapping feature names to importance weights
        """
        self.feature_names = feature_names
        
        # Apply importance weights if provided
        self.importance_weights = np.ones(features.shape[1])
        if importance_weights and feature_names:
            for i, name in enumerate(feature_names):
                if name in importance_weights:
                    self.importance_weights[i] = importance_weights[name]
        
        # Apply weights to features
        weighted_features = features * self.importance_weights
        
        # Scale the features
        self.scaled_features = self.scaler.fit_transform(weighted_features)
        self.model.fit(self.scaled_features)
        self.is_fit = True
        
    def recommend(self, 
                 query_property: Dict[str, Any], 
                 property_data: List[Dict[str, Any]], 
                 query_features: np.ndarray,
                 filter_by_location: bool = True,
                 location_params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend similar properties based on a query property.
        
        Args:
            query_property: Property to find similar properties for
            property_data: List of all property data
            query_features: Preprocessed features for query property
            filter_by_location: Whether to filter results by location
            location_params: Location parameters for filtering (city, province, etc.)
            
        Returns:
            List of recommended properties sorted by similarity
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        # Apply importance weights to query features
        weighted_query = query_features * self.importance_weights
            
        # Scale the query features
        scaled_query = self.scaler.transform(weighted_query.reshape(1, -1))
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(scaled_query)
        
        # Get the recommended properties
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx < len(property_data):  # Safety check
                rec = property_data[idx].copy()  # Copy to avoid modifying original data
                rec['similarity_score'] = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
                
                # Apply location filtering if enabled
                if filter_by_location and location_params:
                    if self._matches_location_criteria(rec, location_params):
                        recommendations.append(rec)
                else:
                    recommendations.append(rec)
            
        return recommendations
    
    def _matches_location_criteria(self, property_dict: Dict[str, Any], 
                                  location_params: Dict[str, str]) -> bool:
        """
        Check if a property matches the specified location criteria.
        
        Args:
            property_dict: Property dictionary
            location_params: Location parameters for filtering
            
        Returns:
            True if property matches location criteria, False otherwise
        """
        subject = property_dict.get('subject', {})
        
        # Extract location information from property
        prop_city = subject.get('city', '') or subject.get('subject_city', '')
        prop_province = subject.get('province', '') or subject.get('subject_province', '')
        prop_postal = subject.get('postal_code', '') or subject.get('subject_postal_code', '')
        prop_municipality = subject.get('municipality_district', '') or subject.get('subject_municipality_district', '')
        
        # Check each criterion if specified
        if 'city' in location_params and location_params['city']:
            if location_params['city'].lower() not in prop_city.lower():
                return False
                
        if 'province' in location_params and location_params['province']:
            if location_params['province'].upper() != prop_province.upper():
                return False
                
        if 'postal_code' in location_params and location_params['postal_code']:
            # Match first 3 characters of postal code (Forward Sortation Area)
            if location_params['postal_code'][:3].upper() != prop_postal[:3].upper():
                return False
                
        if 'municipality' in location_params and location_params['municipality']:
            if location_params['municipality'].lower() not in prop_municipality.lower():
                return False
                
        return True
    
    def recommend_batch(self, 
                       query_properties: List[Dict[str, Any]], 
                       property_data: List[Dict[str, Any]],
                       query_features: np.ndarray,
                       filter_by_location: bool = True,
                       location_params: Optional[Dict[str, str]] = None) -> List[List[Dict[str, Any]]]:
        """
        Recommend similar properties for a batch of query properties.
        
        Args:
            query_properties: List of properties to find similar properties for
            property_data: List of all property data
            query_features: Preprocessed features for query properties
            filter_by_location: Whether to filter results by location
            location_params: Location parameters for filtering
            
        Returns:
            List of lists of recommended properties
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        # Apply importance weights to query features
        weighted_queries = query_features * self.importance_weights
            
        # Scale the query features
        scaled_queries = self.scaler.transform(weighted_queries)
        
        # Find nearest neighbors for all queries
        distances, indices = self.model.kneighbors(scaled_queries)
        
        # Get the recommended properties for each query
        all_recommendations = []
        for i in range(len(query_properties)):
            recommendations = []
            for j, idx in enumerate(indices[i]):
                if idx < len(property_data):  # Safety check
                    rec = property_data[idx].copy()
                    rec['similarity_score'] = 1.0 / (1.0 + distances[i][j])
                    
                    # Apply location filtering if enabled
                    if filter_by_location and location_params:
                        if self._matches_location_criteria(rec, location_params):
                            recommendations.append(rec)
                    else:
                        recommendations.append(rec)
            
            all_recommendations.append(recommendations)
            
        return all_recommendations
    
    def explain_recommendations(self, 
                              query_property: Dict[str, Any], 
                              recommendations: List[Dict[str, Any]],
                              top_n_features: int = 5) -> List[Dict[str, Dict[str, float]]]:
        """
        Explain why properties were recommended by showing feature importance.
        
        Args:
            query_property: The query property
            recommendations: List of recommended properties
            top_n_features: Number of top features to include in explanation
            
        Returns:
            List of dictionaries with feature contributions for each recommendation
        """
        if not self.feature_names or not self.is_fit:
            return []
            
        explanations = []
        
        # Extract query features
        query_subject = query_property.get('subject', {})
        query_features_dict = {}
        for name in self.feature_names:
            if name.startswith('subject_'):
                key = name[8:]  # Remove 'subject_' prefix
                query_features_dict[name] = query_subject.get(key, 0)
            else:
                query_features_dict[name] = query_property.get(name, 0)
        
        for rec in recommendations:
            rec_subject = rec.get('subject', {})
            rec_features_dict = {}
            for name in self.feature_names:
                if name.startswith('subject_'):
                    key = name[8:]  # Remove 'subject_' prefix
                    rec_features_dict[name] = rec_subject.get(key, 0)
                else:
                    rec_features_dict[name] = rec.get(name, 0)
            
            # Calculate feature differences
            feature_diffs = {}
            for name in self.feature_names:
                query_val = query_features_dict.get(name, 0)
                rec_val = rec_features_dict.get(name, 0)
                
                # Handle categorical features (one-hot encoded)
                if '_' in name and name.split('_')[-1] in ['True', 'False']:
                    feature_diffs[name] = 0 if query_val == rec_val else 1
                else:
                    # Calculate normalized difference for numerical features
                    try:
                        query_val = float(query_val)
                        rec_val = float(rec_val)
                        # Avoid division by zero
                        max_val = max(abs(query_val), abs(rec_val))
                        if max_val > 0:
                            feature_diffs[name] = abs(query_val - rec_val) / max_val
                        else:
                            feature_diffs[name] = 0
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        feature_diffs[name] = 0 if query_val == rec_val else 1
            
            # Apply importance weights if available
            if self.importance_weights is not None and len(self.importance_weights) == len(self.feature_names):
                for i, name in enumerate(self.feature_names):
                    feature_diffs[name] *= self.importance_weights[i]
            
            # Get top contributing features
            top_features = sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
            
            # Normalize to get relative importance
            total = sum(diff for _, diff in top_features)
            if total > 0:
                importance = {name: diff/total for name, diff in top_features}
            else:
                importance = {name: 1.0/len(top_features) for name, _ in top_features}
            
            explanations.append({
                'orderID': rec.get('orderID'),
                'feature_importance': importance
            })
            
        return explanations
    
    def set_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """
        Set importance weights for features.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance values
        """
        if not self.feature_names:
            raise ValueError("Feature names not available. Fit the model first.")
            
        weights = np.ones(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name in importance_dict:
                weights[i] = importance_dict[name]
                
        self.importance_weights = weights 