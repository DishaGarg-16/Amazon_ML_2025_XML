# Product Price Prediction - Smart Product Pricing Challenge

## Overview
This project targets the **Smart Product Pricing Challenge** (e.g., Amazon ML Challenge 2025), where the goal is to build a machine learning model that analyzes product details to predict optimal prices. The relationship between product attributes—such as brand, specifications, and pack quantity—and pricing is complex. This solution analyzes these details holistically to suggest a price.

**Performance:**
- **Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)
- **Achieved Score:** **50.02** (using the Text-Only LightGBM approach)

## Dataset details
The dataset consists of 75,000 training samples and 75,000 test samples with the following fields:
- **sample_id:** Unique identifier.
- **catalog_content:** Text field containing title, product description, and Item Pack Quantity (IPQ).
- **image_link:** Public URL for product images.
- **price:** Target variable (training data only).

## Methodology

### 1. Data Cleaning & Feature Engineering
We implemented a robust extraction pipeline to handle noisy text data, focusing on attributes known to influence price:
- **Quantity Extraction:** Regex patterns were used to parse `unit_volume`, `pack_count`, and `unit_measure` from `catalog_content`.
- **Derived Features:** `total_quantity` (`unit_vol` * `pack_count`) was calculated to capture the "bulk" factor.
- **Brand Parsing:** Extracted brand names, grouping less frequent brands into 'OTHER' to manage high cardinality.
- **Handling Missing Data:** Numerical values imputed with median; categorical values One-Hot Encoded.

### 2. Primary Approach (Submitted - SMAPE 50.02)
The core model focuses on text and structured metadata. This approach yielded the best leaderboard performance.

- **Text Processing:** Used `TfidfVectorizer` (1-2 n-grams, 10k max features) to capture semantic signals from the `catalog_content`.
- **Structured Data:** Concatenated encoded features (Brand, Quantity) with text vectors.
- **Model:** **LightGBM Regressor**
    - **Optimization:** Trained on `log1p(price)` to handle the skewed price distribution.
    - **Post-Processing:** Inverse transform (`expm1`) and safety clipping to ensure all predictions are positive floats.

### 3. Experimental Approach (Multimodal Ensemble)
Following the challenge tips to "consider both textual and visual features," we experimented with a multimodal approach. While not the final submission, it demonstrates advanced capability.

- **Image Feature Extraction:**
    - Downloaded images from provided links.
    - Extracted embeddings using a pre-trained **ResNet18** (ImageNet weights), creating a 512-dimensional vector per product.
- **Ensemble Strategy:**
    - Trained a secondary LightGBM model on image embeddings.
    - **Blending:** Combined Text-Model and Image-Model predictions using Ridge Regression.
    - *Note:* This approach requires significant compute for image downloading and processing.

## Constraints & Integrity
- **No External Data:** The model relies solely on the provided dataset; no external scraping or API lookups were used.
- **Output Format:** Predictions are strictly positive float values formatted to 2 decimal places in a CSV file.

## Requirements
- `pandas`, `numpy`, `scikit-learn`
- `lightgbm`
- `torch`, `torchvision` (for image features)
- `transformers` (optional dependency imported in code)

