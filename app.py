from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- NLTK Setup ---
def download_nltk_data(resource_id, resource_path):
    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource_id}' found.")
    except LookupError:
        print(f"NLTK resource '{resource_id}' not found. Downloading...")
        nltk.download(resource_id)

download_nltk_data('stopwords', 'corpora/stopwords')
download_nltk_data('punkt', 'tokenizers/punkt')
download_nltk_data('wordnet', 'corpora/wordnet')
download_nltk_data('omw-1.4', 'corpora/omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Load Saved Artifacts ---
try:
    model = joblib.load('xgb_fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    one_hot_encoder = joblib.load('one_hot_encoder.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    with open('significant_features.pkl', 'rb') as f:
        significant_features = pickle.load(f)
    with open('optimal_threshold.txt', 'r') as f:
        optimal_threshold = float(f.read())
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}. Make sure all .pkl and .txt files are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during artifact loading: {e}")
    exit()


# --- Text Processing Functions (from ML.py) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Location Processing Function (from ML.py) ---
def split_location(location):
    # Handles splitting and basic cleaning for country, state, city
    if pd.isna(location) or location == 'Unknown':
        return pd.Series({'country': 'Unknown', 'state': 'Unknown', 'city': 'Unknown'})
    parts = location.split(', ')
    parts.extend(['Unknown'] * (3 - len(parts)))
    country, state, city = parts[:3]
    country = country.strip() if country.strip() else 'Unknown'
    state = state.strip() if state.strip() else 'Unknown'
    city = city.strip() if city.strip() else 'Unknown'
    # State cleaning (must be alpha, len >= 2)
    state = state if state == 'Unknown' or (state.isalpha() and len(state) >= 2) else 'Unknown'
    # City cleaning (allow letters, spaces, hyphens)
    city = city if city == 'Unknown' or all(c.isalpha() or c in [' ', '-'] for c in city) else 'Unknown'
    return pd.Series({'country': country, 'state': state, 'city': city})


# --- Feature Engineering Pipeline Function ---
def transform_input(data):
    """
    Applies the full feature engineering pipeline matching ML.py to raw input data.
    Args:
        data (dict): A dictionary representing a single job posting.
    Returns:
        pd.DataFrame: A single row DataFrame with features ready for scaling.
                     Returns None if a critical error occurs.
    """
    try:
        # Define column groups based on ML.py logic
        original_binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
        missing_indicator_source_cols = [
            'company_profile', 'requirements', 'benefits', 'employment_type',
            'required_experience', 'required_education', 'industry', 'function',
            'location', 'department', 'salary_range'
        ]
        text_cols_to_process = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        low_card_cols_ohe = ['employment_type', 'required_experience']
        high_card_cols_target = ['required_education', 'industry', 'function', 'country', 'state', 'city']

        # --- Start Processing ---
        df_input = pd.DataFrame([data]) # Work with a DataFrame row

        # 1. Handle Original Binary Features
        for col in original_binary_cols:
            df_input[col] = int(data.get(col, 0)) # Default to 0 if missing

        # 2. Create Missingness Indicators & Store Original Values for Imputation Later
        original_values_for_impute = {}
        for col in missing_indicator_source_cols:
            value = data.get(col) # Get original value
            is_missing = pd.isna(value) or value == '' or str(value).lower() == 'unknown'
            df_input[f'{col}_missing'] = 1 if is_missing else 0
            original_values_for_impute[col] = 'Unknown' if is_missing else value # Store 'Unknown' or original value

        # 3. Create 'has_salary_range' (based on missingness of original salary_range input)
        df_input['has_salary_range'] = 1 - df_input['salary_range_missing']

        # 4. Location Processing
        location_val = original_values_for_impute.get('location', 'Unknown') # Use potentially imputed value
        loc_split = split_location(location_val)
        df_input['country'] = loc_split['country']
        df_input['state'] = loc_split['state']
        df_input['city'] = loc_split['city']

        # 5. Text Processing (using original or imputed 'Unknown' values)
        df_input['combined_text'] = ''
        for col in text_cols_to_process:
            text_val = original_values_for_impute.get(col, '') # Use stored value ('Unknown' or original)
            cleaned = clean_text(text_val)
            preprocessed = preprocess_text(cleaned)
            df_input['combined_text'] += preprocessed + ' '
        df_input['combined_text'] = df_input['combined_text'].str.strip()

        # Apply TF-IDF using the loaded vectorizer
        tfidf_features_sparse = tfidf_vectorizer.transform(df_input['combined_text'])
        tfidf_df = pd.DataFrame(tfidf_features_sparse.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # 6. Prepare inputs for Encoders (using original or imputed 'Unknown' values)
        df_for_ohe = pd.DataFrame()
        for col in low_card_cols_ohe:
             df_for_ohe[col] = [original_values_for_impute.get(col, 'Unknown')]

        df_for_target = pd.DataFrame()
        for col in high_card_cols_target:
             # Use values derived from location split for country/state/city
             if col in ['country', 'state', 'city']:
                  df_for_target[col] = df_input[col].astype(str) # Already processed
             else:
                  df_for_target[col] = [str(original_values_for_impute.get(col, 'Unknown'))] # Use stored value

        # Apply Encoders using loaded objects (use transform, NOT fit_transform)
        one_hot_encoded = one_hot_encoder.transform(df_for_ohe)
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(low_card_cols_ohe))

        target_encoded = target_encoder.transform(df_for_target) # TargetEncoder needs target in training, but not for transform
        target_encoded_df = pd.DataFrame(target_encoded, columns=high_card_cols_target)

        # 7. Combine Base + Encoded Features
        # Select the columns that remained in `df` in ML.py before concatenations
        # These are: original binaries + missing indicators + has_salary_range
        base_cols = original_binary_cols + [f'{col}_missing' for col in missing_indicator_source_cols] + ['has_salary_range']
        df_base = df_input[base_cols].copy() # Select only these relevant columns

        df_base_plus_encoded = pd.concat([df_base.reset_index(drop=True),
                                          one_hot_df.reset_index(drop=True),
                                          target_encoded_df.reset_index(drop=True)], axis=1)

        # 8. Remove TF-IDF Overlaps
        base_plus_encoded_cols_list = df_base_plus_encoded.columns.tolist()
        overlap_cols = [col for col in tfidf_df.columns if col in base_plus_encoded_cols_list]
        if overlap_cols:
            tfidf_df_cleaned = tfidf_df.drop(columns=overlap_cols)
        else:
            tfidf_df_cleaned = tfidf_df # No overlap

        # 9. Final Concatenation
        df_combined = pd.concat([df_base_plus_encoded.reset_index(drop=True),
                                 tfidf_df_cleaned.reset_index(drop=True)], axis=1)

        # 10. Feature Selection (using loaded significant_features list)
        # Ensure all significant features exist, add missing ones with 0
        current_cols = df_combined.columns
        for feat in significant_features:
            if feat not in current_cols:
                df_combined[feat] = 0 # Add missing feature column with 0

        # Select ONLY the significant features IN THE CORRECT ORDER defined during training
        df_final_features = df_combined[significant_features]

        # Final sanity check for shape
        if df_final_features.shape[1] != len(significant_features):
             raise ValueError(f"Final feature shape mismatch! Expected {len(significant_features)}, got {df_final_features.shape[1]}")

        return df_final_features

    except Exception as e:
        print(f"Error during data transformation: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Flask App Initialization & Routes (Keep the same as before) ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        print("Received data:", data) # Debugging

        features_df = transform_input(data)

        if features_df is None:
             return render_template('result.html', prediction_text="Error during data processing.", probability_text="")

        print("Shape of features for scaling:", features_df.shape) # Debugging

        # Check shape before scaling
        if features_df.shape[1] != scaler.n_features_in_:
             return render_template('result.html', prediction_text=f"Error: Feature shape mismatch before scaling. Expected {scaler.n_features_in_}, got {features_df.shape[1]}.", probability_text="")

        scaled_features = scaler.transform(features_df)
        print("Features scaled successfully.") # Debugging

        probability = model.predict_proba(scaled_features)[0][1]
        prediction = (probability >= optimal_threshold).astype(int)
        print(f"Prediction probability: {probability:.4f}, Threshold: {optimal_threshold}, Prediction: {prediction}") # Debugging

        prediction_text = 'Fraudulent Job Posting' if prediction == 1 else 'Legitimate Job Posting'
        probability_text = f'{probability * 100:.2f}% likely to be Fraudulent'

        return render_template('result.html', prediction_text=prediction_text, probability_text=probability_text)

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', prediction_text=f'Error during prediction: {e}', probability_text="")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Turn debug=False for production