# Property Recommendation System

A system for recommending similar properties (comps) based on appraisal data, implemented using four different methods:

1. **KNN (K-Nearest Neighbors)** - Uses traditional distance-based similarity
2. **Neural Network** - Uses a siamese network architecture to learn embeddings
3. **LLM** - Uses language model embeddings for semantic similarity
4. **LLAMA** - Uses fine-tuned LLAMA model with Unsloth for optimized property recommendations

## Project Structure

```
Property-Recommendation-System/
├── data/
│   └── appraisals_dataset.json   # Appraisal dataset
├── results/                      # Output directory for recommendations
├── cache/                        # Cache directory for models and embeddings
├── src/
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── knn_recommender.py        # KNN-based recommender
│   ├── neural_recommender.py     # Neural network-based recommender
│   ├── llm_recommender.py        # LLM-based recommender
│   ├── llama_recommender.py      # LLAMA-based recommender with Unsloth
│   ├── main.py                   # Main script to run the system
│   └── app.py                    # Streamlit web application
└── requirements.txt              # Project dependencies
```

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you want to use the LLAMA-based recommender, run the setup script:

```bash
# Install and download the LLAMA model (this may take a while)
python setup_llama.py

# Use a specific model and cache directory
python setup_llama.py --model unsloth/llama-3-8b-bnb-4bit --cache_dir cache/llama

# Skip dependency installation
python setup_llama.py --skip_deps

# Skip model testing
python setup_llama.py --skip_test
```

## Usage

Run the recommendation system using the following command:

```bash
# Run all recommendation methods
python src/main.py --data_path data/appraisals_dataset.json

# Run a specific method (knn, neural, llm, or llama)
python src/main.py --method llama

# Use a specific property as query
python src/main.py --query_id "4762597"

# Customize parameters
python src/main.py --knn_neighbors 10 --nn_epochs 50 --llm_model "all-MiniLM-L6-v2" --llama_model "unsloth/llama-3-8b-bnb-4bit"
```

## Methods

### 1. KNN-based Recommender

Uses K-Nearest Neighbors algorithm to find properties with similar features. Features are preprocessed and scaled before similarity computation.

### 2. Neural Network-based Recommender

Uses a siamese neural network with triplet loss to learn embeddings for properties. Similar properties will have embeddings that are close in the embedding space.

### 3. LLM-based Recommender

Uses sentence transformers to generate embeddings for textual descriptions of properties. Computes similarity in the embedding space to find similar properties.

### 4. LLAMA-based Recommender

Uses a LLAMA model fine-tuned with Unsloth to generate property recommendations based on detailed property descriptions. The model is fine-tuned using LoRA adapters on the appraisal dataset to learn what makes properties comparable for valuation purposes.

#### LLAMA Fine-tuning Process

The LLAMA recommender:
1. Converts property data into instruction-response pairs based on existing comps
2. Fine-tunes a LLAMA model using Unsloth's optimization techniques and LoRA adapters
3. Uses the fine-tuned model to generate recommendations for new properties
4. Can provide detailed explanations for why properties are comparable

## Evaluation

To evaluate the performance of the recommenders, use the `--evaluate` flag:

```bash
python src/main.py --evaluate
```

## Web Application

The system also includes a Streamlit web application for easier interaction:

```bash
streamlit run src/app.py
```

## Extending the System

To extend the system with new methods or features:

1. Add a new recommender class in a new file
2. Implement the necessary methods (fit, recommend, etc.)
3. Update `main.py` and `app.py` to include the new method

## Dependencies

- numpy, pandas, scikit-learn
- tensorflow
- sentence-transformers
- torch, transformers
- unsloth, peft (for LLAMA fine-tuning)
- streamlit (for web app) 