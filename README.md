# Property Recommendation System

This project implements a Python script (`property_similarity.py`) to recommend comparable properties (comps) for a subject property based on an appraisal dataset. It aims to replicate an appraiser's choice of comps by using a k-Nearest Neighbors (k-NN) machine learning algorithm. To run the app + the front end, please install all dependencies and then "run streamlit run app.py" in your CLI.

## Table of Contents
1.  [Purpose](#purpose)
2.  [Methodology](#methodology)
    *   [Data Standardization](#data-standardization)
    *   [Feature Engineering](#feature-engineering)
    *   [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)
    *   [Candidate Filtering & Data Override](#candidate-filtering--data-override)
3.  [Project Structure](#project-structure)
4.  [Setup and Execution](#setup-and-execution)
    *   [Prerequisites](#prerequisites)
    *   [Running the Script](#running-the-script)
5.  [Input Data Format](#input-data-format)
6.  [Development Journey & Key Learnings](#development-journey--key-learnings)
    *   [Initial Feature Parsing](#initial-feature-parsing)
    *   [Debugging GLA and Lot Size](#debugging-gla-and-lot-size)
    *   [Age Calculation Nuances](#age-calculation-nuances)
    *   [Structure Type Normalization](#structure-type-normalization)
    *   [Addressing Data Discrepancies for Actual Comps](#addressing-data-discrepancies-for-actual-comps)
    *   [Limitations](#limitations)
7.  [Potential Future Improvements](#potential-future-improvements)

## Purpose

The primary goal is to analyze a subject property from an appraisal report and, from a list of ~100 other property listings, identify the top N (typically 3) properties that are most similar to the subject property. This simulates the process an appraiser goes through when selecting comparable sales.

## Methodology

The script employs several techniques to achieve its goal:

### Data Standardization
A crucial step involves standardizing property features from various sources (subject property, appraiser-chosen comps, and candidate property listings). This includes:
*   Parsing numeric values like GLA (Gross Living Area), lot size, and prices from string formats.
*   Calculating property age based on `year_built` and a `reference_year` (derived from the appraisal's effective date).
*   Standardizing categorical features like `structure_type`, `style`, and `condition`.
*   Calculating `total_baths` from full and half bath counts.
*   Calculating `distance_to_subject_km` using Haversine distance for listings (if lat/lon available) or using provided distances for appraiser comps.

### Feature Engineering
Selected numerical and categorical features are used for the k-NN model.
*   **Numerical Features Used:** `gla`, `age`, `lot_size_sf`, `bedrooms`, `total_baths`, `distance_to_subject_km`, `basement_sqft`, `room_count`.
*   **Categorical Features Used:** `structure_type`, `style`, `condition`, `cooling_type`, `heating_type`, `primary_exterior_finish`.

### k-Nearest Neighbors (k-NN)
*   **Preprocessing:**
    *   Numerical features are imputed using the median and then scaled to a 0-1 range using `MinMaxScaler`.
    *   Categorical features are imputed with a constant "Unknown" value and then one-hot encoded.
*   **Model:** A `NearestNeighbors` model from `scikit-learn` with a Euclidean distance metric is used to find the most similar candidate properties to the subject property.

### Candidate Filtering & Data Override
1.  **Structure Type Pre-filtering:** Before running k-NN, the pool of candidate properties is filtered to include only those with the same `structure_type` as the subject property. This significantly narrows down the search space to more relevant candidates.
2.  **Data Override for Known Comps:** If a candidate property (from the listings) is also identified as one of the appraiser's actual chosen comps (by matching address), its features (e.g., GLA, age, distance, condition) are overridden with the (presumably more accurate or relevant for comparison) standardized data from the appraiser's comp information. This step was key to improving the match rate with the appraiser's choices.

## Project Structure

```
Property-Recommendation-System/
├── property_similarity.py    # Main Python script
├── appraisals_dataset.json   # Input appraisal data (not committed, user provided)
├── README.md                 # This file
└── requirements.txt          # Python dependencies (to be created)
```

## Setup and Execution

### Prerequisites
*   Python 3.x
*   The necessary Python packages can be installed using pip. A `requirements.txt` should be created:
    ```
    pandas
    numpy
    scikit-learn
    ```
    Install them using:
    `pip install -r requirements.txt`

### Running the Script
1.  Ensure you have the `appraisals_dataset.json` file in the same directory as `property_similarity.py`.
2.  Run the script from the command line:
    `python property_similarity.py`

The script will process the first appraisal in the dataset, print summaries of the subject property, actual appraiser comps, and the algorithm's recommended comps, including a comparison of how many actual comps were matched.

## Input Data Format

The script expects an `appraisals_dataset.json` file with a structure similar to this (simplified):

```json
{
  "appraisals": [
    {
      "orderID": "unique_appraisal_id",
      "subject": {
        "address": "123 Main St",
        "gla": "1500 SqFt",
        "year_built": "1990",
        "effective_date": "Jan/01/2023",
        // ... other subject features (num_beds, num_baths, style, condition, etc.)
        // "latitude", "longitude" (optional, if not present, demo values are used)
      },
      "comps": [ // Appraiser's chosen comparables
        {
          "address": "125 Main St",
          "gla": "1550",
          "year_built": "1992",
          "sale_price": "500000",
          "distance_to_subject": "0.5 KM",
          // ... other comp features
        }
        // ... more comps
      ],
      "properties": [ // List of ~100 candidate properties
        {
          "id": "prop_id_1",
          "address": "10 Nearby Rd",
          "gla": 1400, // Can be int or string
          "year_built": 30, // Can be year or age
          "latitude": 44.25,
          "longitude": -76.58,
          // ... other listing features (bedrooms, bathrooms, structure_type, property_sub_type, etc.)
        }
        // ... more properties
      ]
    }
    // ... more appraisals
  ]
}
```
The script currently processes only the first appraisal in the `appraisals` list. Latitude and longitude for the subject property are hardcoded for demonstration but would ideally be geocoded or provided.

## Development Journey & Key Learnings

The development of this script was an iterative process focused on data cleaning, feature engineering, and refining the similarity logic:

### Initial Feature Parsing
*   Basic helper functions were created to clean strings, parse square footage (`parse_sq_ft`), prices (`parse_price`), and bathroom counts (`parse_bath_count`).

### Debugging GLA and Lot Size
*   Initial runs showed `gla` and `lot_size_sf` were not being parsed correctly (resulting in `None`). This was traced to issues in the `parse_sq_ft` regex and input data types.
*   Intensive debugging involved adding detailed print statements within `parse_sq_ft` to inspect input values, cleaning steps, and regex matching.
*   The solution was to simplify `parse_sq_ft` to attempt direct `float()` conversion after basic cleaning (removing "sqft", commas), which proved more robust for the given data than complex regex alone.

### Age Calculation Nuances
*   Calculating `age` required careful handling of `year_built` (which could be a year or an age for listings) and `reference_year` from the appraisal's `effective_date`.
*   Logic was added to `standardize_property_features` to differentiate between `subject`, `comp`, and `property_listing` data types, as they had different field names or formats for age/year built.
*   Heuristics were implemented to detect and correct potentially swapped `age` and `year_built` values in listings.
*   The subject property's `year_built` parsing was particularly tricky, requiring fallback mechanisms (digit extraction) when standard regex failed, possibly due to hidden characters or unusual string formats not caught by `.strip()`.

### Structure Type Normalization
*   A pre-filtering step based on `structure_type` was introduced to narrow down candidates.
*   This revealed inconsistencies in how `structure_type` (e.g., "Freehold Townhouse", "Condo Townhouse") was represented in listings versus the subject/comps.
*   Normalization logic was added to map variations like "Condo Townhouse" to a standard "Townhouse" to ensure correct filtering. This dramatically improved the relevance of the candidate pool.

### Addressing Data Discrepancies for Actual Comps
*   Even with improved features and filtering, matching all appraiser comps was challenging.
*   **Key Finding 1: Missing Candidate Data:** One appraiser comp (`930 Amberdale Cres`) was consistently not found because it was determined to be missing from the `properties` array (the candidate pool) for the specific appraisal in the dataset. The script cannot select what it's not given.
*   **Key Finding 2: Feature Discrepancies:** For comps that *were* in both the appraiser's list and the candidate listings, their features (especially `gla`, `age`, `condition`, and crucially `distance_to_subject_km`) often differed. The k-NN used the listing data, which could be less accurate or represent a different snapshot than the appraiser's data.
*   **Solution Implemented:** A crucial improvement was to add a step that *overrides* the features of a candidate listing with the features from the appraiser's actual comp data if an address match is found. This ensured that for these known good comps, the k-NN used the more reliable appraiser-defined characteristics, leading to a successful match of 2 out of 3 comps.

### Limitations
*   **Data Quality is Paramount:** The accuracy of the k-NN recommendations is highly dependent on the quality, completeness, and consistency of the input `properties` data. Inaccurate lat/lon leading to wrong distances, or different GLA/age values compared to appraiser data, directly impacts results.
*   **Missing Candidate Properties:** If an appraiser's comp is not present in the `properties` list provided for the appraisal, the script cannot recommend it.
*   **Feature Engineering and Weighting:** The current feature set and `MinMaxScaler` treat all features somewhat equally after scaling. More domain knowledge could inform specific feature weighting or more advanced distance metrics (e.g., Gower distance for mixed data types without extensive one-hot encoding).
*   **Geocoding:** The script uses hardcoded subject lat/lon for demo purposes. Real-world application would require robust geocoding for subject and potentially for listings if their lat/lon is missing or suspect.

## Potential Future Improvements
*   **Dynamic Geocoding:** Integrate a geocoding service (e.g., geopy with Nominatim or a paid service) to look up latitudes and longitudes for addresses.
*   **Advanced Feature Weighting/Selection:** Experiment with techniques to assign different weights to features based on their importance (e.g., distance might be more critical than style).
*   **Alternative Similarity Metrics:** Explore metrics like Gower distance, which can handle mixed data types natively and might be more robust to outliers or feature scaling issues.
*   **Hyperparameter Tuning:** Tune `n_neighbors` for the k-NN model or explore other k-NN parameters.
*   **Handling Multiple Appraisals:** Extend the script to process all appraisals in the dataset, not just the first one.
*   **User Interface:** Develop a simple UI (e.g., using Streamlit or Flask) for easier interaction.
*   **Robust Error Handling for Data Inconsistencies:** Add more checks for unexpected data formats or missing critical fields.
*   **Model Evaluation Framework:** Develop a more formal way to evaluate the "correctness" of recommendations beyond simple address matching, perhaps by looking at feature similarity scores.

This project demonstrates a practical approach to building a comparable property recommendation system, highlighting the importance of thorough data standardization and the iterative nature of developing machine learning solutions, especially when dealing with real-world data inconsistencies. 