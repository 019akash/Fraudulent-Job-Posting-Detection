# Fraudulent Job Posting Detection using Machine Learning

This project develops a machine learning model to detect fraudulent job postings, enhanced by statistical validation techniques. It includes a complete pipeline from data preprocessing and feature engineering to model training, evaluation, and deployment via a prototype Flask web application.

## Table of Contents

*   [Project Description](#project-description)
*   [Features](#features)
*   [Technology Stack](#technology-stack)
*   [Project Structure](#project-structure)
*   [Dataset](#dataset)
*   [Methodology Overview](#methodology-overview)
*   [Setup and Installation](#setup-and-installation)
*   [Usage](#usage)
    *   [Running the Training Pipeline](#running-the-training-pipeline)
    *   [Running the Flask Web Application](#running-the-flask-web-application)
*   [Key Results](#key-results)
*   [Future Work](#future-work)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## Project Description

The proliferation of online job platforms has also led to an increase in fraudulent job postings. These scams pose risks such as financial loss, identity theft, and erosion of trust in recruitment platforms. This project aims to build and rigorously validate a machine learning system to automatically identify such fraudulent postings. The system leverages feature engineering, handles class imbalance, compares various ML models, and integrates statistical methods for robust validation and interpretation. A prototype Flask application demonstrates the practical deployment of the final model.

## Features

*   **Data Preprocessing:** Cleaning and preparation of raw job posting data.
*   **Feature Engineering:** Creation of informative features from text (TF-IDF), categorical attributes (One-Hot & Target Encoding), and missing data patterns.
*   **Statistical Validation:** Chi-Square tests, Correlation Analysis, Mann-Whitney U tests, and McNemar's test integrated into the pipeline.
*   **Class Imbalance Handling:** Implemented SMOTENC to address the skewed dataset.
*   **Model Training & Comparison:** Evaluated Logistic Regression, Random Forest, XGBoost, and LightGBM.
*   **Hyperparameter Tuning:** Used GridSearchCV for XGBoost optimization.
*   **Threshold Optimization:** Determined optimal classification thresholds to balance precision and recall based on project goals.
*   **Flask Web Application:** A prototype application for users to input job details and get a fraud prediction.
*   **Saved Artifacts:** Pre-trained model, scaler, encoders, vectorizer, and feature list for consistent deployment.

## Technology Stack

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas: Data manipulation and analysis.
    *   NumPy: Numerical operations.
    *   Scikit-learn: Machine learning (Logistic Regression, Random Forest, OHE, TF-IDF, StandardScaler, GridSearchCV, metrics).
    *   XGBoost: Gradient Boosting model.
    *   LightGBM: Gradient Boosting model.
    *   NLTK: Natural Language Processing (tokenization, stopwords, lemmatization).
    *   Category Encoders: For Target Encoding.
    *   Imbalanced-learn: For SMOTENC.
    *   SciPy: For statistical tests (Chi-Square, Mann-Whitney U).
    *   Joblib & Pickle: For saving and loading model artifacts.
*   **Web Framework:** Flask (for the prototype application).
*   **Frontend:** HTML, CSS.


## Dataset

The project utilizes the **EMSCAD dataset** from the Laboratory of Information & Communication Systems Security at the University of the Aegean. It contains approximately 17,880 job postings labeled as fraudulent or legitimate.

*   **Source:** `http://emscad.samos.aegean.gr/`

## Methodology Overview

1.  **Data Collection & Preprocessing:** Loaded the dataset, performed initial cleaning (binary format conversion), handled missing values by creating indicator features and imputing with "Unknown".
2.  **Feature Engineering:**
    *   **Location:** Parsed `location` into `country`, `state`, `city`.
    *   **Text:** Cleaned text, applied lemmatization, and used TF-IDF to create 8000 text features.
    *   **Categorical:** Used One-Hot Encoding for low-cardinality and Target Encoding for high-cardinality features.
3.  **Feature Selection:** Applied Lasso (L1) regression, followed by Mann-Whitney U tests for statistical validation, resulting in 482 significant features.
4.  **Class Imbalance:** Addressed using SMOTENC on the training data.
5.  **Model Training & Evaluation:** Trained Logistic Regression, Random Forest, XGBoost, and LightGBM. Evaluated using PR-AUC, F1-score, Precision, and Recall, focusing on the fraudulent class. Optimized classification thresholds.
6.  **Hyperparameter Tuning:** Fine-tuned XGBoost using GridSearchCV.
7.  **Statistical Validation:** Incorporated Chi-Square tests, Correlation analysis, and McNemar's test.
8.  **Deployment:** Developed a prototype Flask web application.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/019akash/fraudulent-job-posting-detection.git
    cd fraudulent-job-posting-detection
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK resources (if running `app.py` or parts of `ML.py` for the first time):**
    The `app.py` script attempts to download necessary NLTK resources (`stopwords`, `punkt`, `wordnet`, `omw-1.4`) if they are not found. You can also run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```
5.  **(If using Git LFS for large model files):**
    Ensure Git LFS is installed (`git lfs install` system-wide once). Then pull LFS files:
    ```bash
    git lfs pull
    ```

## Usage

### Running the Training Pipeline (Optional - for reproducibility)

If you wish to re-train the model and regenerate the artifacts:
1.  Ensure the `Dataset.csv` is present in the root directory (or update the path in `ML.py`).
2.  Run the `ML.py` script:
    ```bash
    python ML.py
    ```
    This script will perform all preprocessing, feature engineering, training, evaluation, and will save the model and other artifacts (`.pkl` files, `optimal_threshold.txt`).

### Running the Flask Web Application

1.  Ensure all required artifacts (`.pkl` files, `optimal_threshold.txt`) are present in the root directory (either from the repo or generated by running `ML.py`).
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000/`.
4.  Fill in the job posting details in the form and click "Predict Fraud".
5.  The application will display the prediction (Fraudulent/Legitimate) and the probability score.

## Key Results

*   The final selected model (Untuned XGBoost with 1:1 SMOTENC, optimal threshold 0.2694) achieved:
    *   **Recall (Fraudulent Class): 95%**
    *   Precision (Fraudulent Class): 89%
    *   F1-Score (Fraudulent Class): 0.92
    *   PR-AUC: 0.9589
*   Tuned XGBoost offered the highest precision (98%) and F1-score (0.94) but with slightly lower recall (91%).
*   The project successfully prioritized recall to minimize missed fraudulent postings.

## Future Work

*   Explore deep learning-based text embeddings (e.g., BERT).
*   Investigate alternative class imbalance techniques.
*   Implement model-agnostic explainability methods (SHAP, LIME).
*   Analyze temporal patterns if time-series data becomes available.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgements

*   The EMSCAD dataset provided by the Laboratory of Information & Communication Systems Security at the University of the Aegean.
