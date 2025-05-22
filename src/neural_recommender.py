import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional


class NeuralPropertyRecommender:
    """
    Neural network-based property recommendation system.
    Uses a siamese network architecture to learn similarity between properties.
    """
    
    def __init__(self, 
                 embedding_dim: int = 64, 
                 hidden_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 activation: str = 'relu'):
        """
        Initialize the neural network-based property recommender.
        
        Args:
            embedding_dim: Dimension of the property embedding
            hidden_units: List of hidden units for each dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            activation: Activation function for hidden layers
        """
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.scaler = StandardScaler()
        self.model = None
        self.embedding_model = None
        self.is_fit = False
        self.feature_names = None
        self.importance_weights = None
        
    def _build_encoder(self, input_shape: Tuple[int]) -> keras.Model:
        """
        Build the encoder part of the siamese network.
        
        Args:
            input_shape: Shape of the input features
            
        Returns:
            Encoder model
        """
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Add dense layers with dropout
        for units in self.hidden_units:
            x = layers.Dense(units, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
        # Final embedding layer
        embeddings = layers.Dense(self.embedding_dim, name="embeddings")(x)
        
        # Create the encoder model
        encoder = keras.Model(inputs=inputs, outputs=embeddings, name="encoder")
        return encoder
    
    def _build_siamese_model(self, input_shape: Tuple[int]) -> Tuple[keras.Model, keras.Model]:
        """
        Build the full siamese model with triplet loss.
        
        Args:
            input_shape: Shape of the input features
            
        Returns:
            Tuple of (full siamese model, encoder model)
        """
        # Build the shared encoder
        encoder = self._build_encoder(input_shape)
        
        # Define the inputs for anchor, positive, and negative
        anchor_input = keras.Input(shape=input_shape, name="anchor")
        positive_input = keras.Input(shape=input_shape, name="positive")
        negative_input = keras.Input(shape=input_shape, name="negative")
        
        # Generate embeddings for each input
        anchor_embedding = encoder(anchor_input)
        positive_embedding = encoder(positive_input)
        negative_embedding = encoder(negative_input)
        
        # TripletLoss layer
        triplet_layer = TripletLossLayer(margin=0.2, name="triplet_loss")(
            [anchor_embedding, positive_embedding, negative_embedding]
        )
        
        # Create the siamese model
        siamese_model = keras.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=triplet_layer,
            name="siamese_network"
        )
        
        # Compile the model
        siamese_model.compile(optimizer=keras.optimizers.Adam(self.learning_rate))
        
        return siamese_model, encoder
    
    def fit(self, 
            features: np.ndarray,
            feature_names: Optional[List[str]] = None,
            importance_weights: Optional[Dict[str, float]] = None,
            triplets: Optional[np.ndarray] = None,
            epochs: int = 20,
            batch_size: int = 64,
            validation_split: float = 0.1,
            early_stopping: bool = True,
            verbose: int = 1) -> keras.callbacks.History:
        """
        Fit the neural network model to the property feature data.
        
        Args:
            features: Feature matrix for properties (n_samples x n_features)
            feature_names: Names of the features
            importance_weights: Dictionary mapping feature names to importance weights
            triplets: Triplets for training (anchor, positive, negative) if available
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            verbose: Verbosity level
            
        Returns:
            Training history
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
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(weighted_features)
        
        # Build the models
        input_shape = (features.shape[1],)
        self.model, self.embedding_model = self._build_siamese_model(input_shape)
        
        # If no triplets are provided, generate them using property similarities
        if triplets is None:
            n_samples = min(10000, features.shape[0])
            triplets = self._generate_improved_triplets(n_samples, features)
        
        # Prepare training data
        anchor_data = self.scaled_features[triplets[:, 0]]
        positive_data = self.scaled_features[triplets[:, 1]]
        negative_data = self.scaled_features[triplets[:, 2]]
        
        # Set up callbacks
        callbacks = []
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss', 
                patience=5, 
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Train the model
        history = self.model.fit(
            [anchor_data, positive_data, negative_data],
            y=np.zeros(len(triplets)),  # Dummy output (triplet loss is computed internally)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fit = True
        return history
    
    def _generate_improved_triplets(self, n_triplets: int, features: np.ndarray) -> np.ndarray:
        """
        Generate improved triplets for training based on feature similarity.
        This creates more meaningful triplets than purely random selection.
        
        Args:
            n_triplets: Number of triplets to generate
            features: Feature matrix for properties
            
        Returns:
            Array of triplet indices (anchor, positive, negative)
        """
        n_samples = features.shape[0]
        triplets = np.zeros((n_triplets, 3), dtype=np.int32)
        
        # Simple feature standardization for similarity calculation
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Calculate pairwise distances between all samples
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(scaled_features, metric='euclidean')
        
        for i in range(n_triplets):
            # Select a random anchor
            anchor_idx = np.random.randint(0, n_samples)
            
            # Get distances from anchor to all other points
            anchor_distances = distances[anchor_idx]
            
            # Sort indices by distance from anchor (excluding anchor itself)
            sorted_indices = np.argsort(anchor_distances)
            sorted_indices = sorted_indices[sorted_indices != anchor_idx]
            
            # Select positive from the closest 20% of points
            positive_candidates = sorted_indices[:max(1, int(0.2 * n_samples))]
            positive_idx = np.random.choice(positive_candidates)
            
            # Select negative from the furthest 20% of points
            negative_candidates = sorted_indices[-max(1, int(0.2 * n_samples)):]
            negative_idx = np.random.choice(negative_candidates)
            
            triplets[i] = [anchor_idx, positive_idx, negative_idx]
            
        return triplets
    
    def get_embeddings(self, features: np.ndarray) -> np.ndarray:
        """
        Get embeddings for property features.
        
        Args:
            features: Feature matrix for properties
            
        Returns:
            Embeddings for the properties
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        # Apply importance weights if available
        if self.importance_weights is not None:
            weighted_features = features * self.importance_weights
        else:
            weighted_features = features
            
        # Scale the features
        scaled_features = self.scaler.transform(weighted_features)
        
        # Get the embeddings
        embeddings = self.embedding_model.predict(scaled_features)
        
        return embeddings
    
    def recommend(self, 
                 query_property: Dict[str, Any], 
                 property_data: List[Dict[str, Any]], 
                 all_features: np.ndarray,
                 query_features: np.ndarray,
                 filter_by_location: bool = True,
                 location_params: Optional[Dict[str, str]] = None,
                 top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend similar properties based on a query property.
        
        Args:
            query_property: Property to find similar properties for
            property_data: List of all property data
            all_features: Features for all properties
            query_features: Features for the query property
            filter_by_location: Whether to filter results by location
            location_params: Location parameters for filtering
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended properties sorted by similarity
        """
        if not self.is_fit:
            raise ValueError("Model has not been fit yet. Call fit() first.")
            
        # Get embeddings for all properties
        all_embeddings = self.get_embeddings(all_features)
        
        # Get embedding for the query property
        query_embedding = self.get_embeddings(query_features.reshape(1, -1))
        
        # Compute cosine similarity between query and all properties
        similarities = self._cosine_similarity(query_embedding, all_embeddings)
        
        # Get indices of top_n similar properties
        top_indices = np.argsort(-similarities)
        
        # Apply location filtering if enabled
        if filter_by_location and location_params:
            filtered_indices = []
            for idx in top_indices:
                if idx < len(property_data):  # Safety check
                    if self._matches_location_criteria(property_data[idx], location_params):
                        filtered_indices.append(idx)
                        if len(filtered_indices) >= top_n:
                            break
            top_indices = np.array(filtered_indices)
        else:
            top_indices = top_indices[:top_n]
        
        # Get the recommended properties
        recommendations = []
        for idx in top_indices:
            if idx < len(property_data):  # Safety check
                rec = property_data[idx].copy()
                rec['similarity_score'] = float(similarities[idx])
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
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            a: First set of vectors (n_samples_a x n_features)
            b: Second set of vectors (n_samples_b x n_features)
            
        Returns:
            Cosine similarity between each pair of vectors
        """
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        # Avoid division by zero
        a_norm[a_norm == 0] = 1e-10
        b_norm[b_norm == 0] = 1e-10
        
        a_normalized = a / a_norm
        b_normalized = b / b_norm
        
        return np.dot(a_normalized, b_normalized.T).flatten()
    
    def explain_recommendations(self, 
                              query_property: Dict[str, Any], 
                              recommendations: List[Dict[str, Any]],
                              top_n_features: int = 5) -> List[Dict[str, Dict[str, Any]]]:
        """
        Generate explanations for recommendations based on feature similarity.
        
        Args:
            query_property: Query property
            recommendations: List of recommended properties
            top_n_features: Number of top features to include in explanation
            
        Returns:
            List of dictionaries with explanation information
        """
        if not self.feature_names:
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
            
            # Calculate feature differences/similarities
            feature_similarities = {}
            for name in self.feature_names:
                query_val = query_features_dict.get(name, 0)
                rec_val = rec_features_dict.get(name, 0)
                
                # Handle categorical features (one-hot encoded)
                if '_' in name and name.split('_')[-1] in ['True', 'False']:
                    feature_similarities[name] = 1.0 if query_val == rec_val else 0.0
                else:
                    # Calculate similarity for numerical features
                    try:
                        query_val = float(query_val)
                        rec_val = float(rec_val)
                        # Calculate similarity (1 - normalized difference)
                        max_val = max(abs(query_val), abs(rec_val))
                        if max_val > 0:
                            feature_similarities[name] = 1.0 - (abs(query_val - rec_val) / max_val)
                        else:
                            feature_similarities[name] = 1.0
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        feature_similarities[name] = 1.0 if query_val == rec_val else 0.0
            
            # Get top contributing features to similarity
            top_features = sorted(feature_similarities.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
            
            # Format the explanation
            explanation = {
                'orderID': rec.get('orderID'),
                'similarity_score': rec.get('similarity_score', 0),
                'similar_features': {name: {'similarity': sim, 'query_value': query_features_dict.get(name), 
                                           'rec_value': rec_features_dict.get(name)} 
                                   for name, sim in top_features}
            }
            
            explanations.append(explanation)
            
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


class TripletLossLayer(layers.Layer):
    """
    Custom layer to compute triplet loss.
    """
    
    def __init__(self, margin: float = 0.2, **kwargs):
        """
        Initialize the triplet loss layer.
        
        Args:
            margin: Margin for triplet loss
            **kwargs: Additional layer parameters
        """
        super(TripletLossLayer, self).__init__(**kwargs)
        self.margin = margin
        
    def call(self, inputs):
        """
        Compute triplet loss.
        
        Args:
            inputs: List of [anchor, positive, negative] embeddings
            
        Returns:
            Triplet loss
        """
        anchor, positive, negative = inputs
        
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Compute triplet loss
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.maximum(basic_loss, 0.0)
        
        # Add as a loss to the model
        self.add_loss(tf.reduce_mean(loss))
        
        # Return dummy output (the loss is added to the model separately)
        return loss 