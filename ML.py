import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import pickle
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib

# Load dataset
df = pd.read_csv("Dataset.csv")

# Convert binary columns from 't'/'f' or '0'/'1' strings to integers (0/1)
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent', 'in_balanced_dataset']
for col in binary_cols:
    # Check unique values to confirm format
    print(f"Unique values in {col} before conversion:", df[col].unique())
    # Replace 't'/'f' with 1/0, and ensure '0'/'1' strings are converted to integers
    df[col] = df[col].replace({'t': 1, 'f': 0, '0': 0, '1': 1}).astype(int)

# Basic info
print("\nDataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive statistics for numerical/binary columns
numerical_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
print("\nDescriptive Statistics for Numerical Columns:")
print(df[numerical_cols].describe())
print("\nSkewness of 'fraudulent':", skew(df['fraudulent']))
print("Kurtosis of 'fraudulent':", kurtosis(df['fraudulent']))

# Frequency distribution for categorical columns
categorical_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
print("\nFrequency Distribution for Categorical Columns:")
for col in categorical_cols:
    print(f"\n{col}:\n", df[col].value_counts(dropna=False, normalize=True))

# Bar plot for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    df[col].value_counts(dropna=False, normalize=True).head(10).plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {col} (Top 10 Categories)')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.show()

# Target variable distribution
print("\nTarget Variable Distribution ('fraudulent'):")
print(df['fraudulent'].value_counts(normalize=True))

# Pie chart for class distribution
plt.figure(figsize=(6, 6))
df['fraudulent'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=['Non-Fraudulent (0)', 'Fraudulent (1)'])
plt.title('Class Distribution of Fraudulent Postings')
plt.ylabel('')
plt.show()

"""-------------------------------------------------------------------------------------------------"""
#Explore Missingness Patterns Statistically
# Create binary indicators for missingness
missing_cols = ['company_profile', 'requirements', 'benefits', 'employment_type', 'required_experience', 'required_education', 'industry', 'function', 'location', 'department', 'salary_range']
for col in missing_cols:
    df[f'{col}_missing'] = df[col].isnull().astype(int)

# Analyze missingness vs. fraudulent
print("\nMissingness vs. Fraudulent (Mean Fraudulent Rate):")
for col in missing_cols:
    print(f"\n{col}:")
    print(df.groupby(f'{col}_missing')['fraudulent'].mean())

# Heatmap of missingness patterns
missingness_df = df[[f'{col}_missing' for col in missing_cols]]
plt.figure(figsize=(10, 6))
sns.heatmap(missingness_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation of Missingness Indicators Across Columns')
plt.show()

# Chi-Square test for association
# Store Chi-Square test results
chi2_results = {}
print("\nChi-Square Tests for Missingness vs. Fraudulent:")
for col in missing_cols:
    contingency_table = pd.crosstab(df[f'{col}_missing'], df['fraudulent'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{col}: Chi2 = {chi2:.4f}, p-value = {p:.4f}, Significant = {p < 0.05}")
    chi2_results[col] = p

# Bar plot of p-values
plt.figure(figsize=(10, 6))
pd.Series(chi2_results).sort_values().plot(kind='bar', color='salmon')
plt.axhline(y=0.05, color='black', linestyle='--', label='p = 0.05')
plt.title('Chi-Square Test p-values: Missingness vs. Fraudulent')
plt.xlabel('Feature')
plt.ylabel('p-value')
plt.legend()
plt.xticks(rotation=45)
plt.show()

"""-----------------------------------------------------------------------------------------------"""
# Step 1: Inspect the location column
print("\nTop 20 Location Values:")
print(df['location'].value_counts(dropna=False).head(20))

# Define a robust split_location function
def split_location(location):
    if pd.isna(location) or location == 'Unknown':
        return pd.Series({'country': 'Unknown', 'state': 'Unknown', 'city': 'Unknown'})
    
    parts = location.split(', ')
    parts.extend(['Unknown'] * (3 - len(parts)))
    country, state, city = parts[:3]
    
    # Clean up empty strings
    country = country if country.strip() else 'Unknown'
    state = state if state.strip() else 'Unknown'
    city = city if city.strip() else 'Unknown'
    
    # Allow spaces and hyphens in city, but ensure it's mostly alphabetic
    if city != 'Unknown' and not all(c.isalpha() or c in [' ', '-'] for c in city):
        city = 'Unknown'
    
    return pd.Series({'country': country, 'state': state, 'city': city})

# Apply the function and create is_remote
df[['country', 'state', 'city']] = df['location'].apply(split_location)
df['is_remote'] = df['location'].str.contains('remote', case=False, na=False).astype(int)

# Clean state and city (handle single characters or non-alpha states)
df['state'] = df['state'].apply(lambda x: 'Unknown' if not x.isalpha() or len(x) <= 1 else x)
df['city'] = df['city'].apply(lambda x: 'Unknown' if not x.isalpha() else x)

# Check the results
print("\nSample of Split Location (First 10 Rows):")
print(df[['location', 'country', 'state', 'city', 'is_remote']].head(10))
print("\nUnique States (Sample):")
print(df['state'].value_counts().head(10))
print("\nUnique Cities (Sample):")
print(df['city'].value_counts().head(10))
print("\nIs Remote Distribution:")
print(df['is_remote'].value_counts(normalize=True))


# Enhance state cleaning to handle single characters and invalid entries
df['state'] = df['state'].apply(lambda x: x if x == 'Unknown' or (x.isalpha() and len(x) >= 2) else 'Unknown')

#Fix city cleaning to allow spaces and hyphens
df['city'] = df['city'].apply(lambda x: x if x == 'Unknown' or all(c.isalpha() or c in [' ', '-'] for c in x) else 'Unknown')

#Re-check unique states and cities after fixing
print("\nUnique States After Enhanced Cleaning (Sample):")
print(df['state'].value_counts().head(10))
print("\nUnique Cities After Fixed Cleaning (Sample):")
print(df['city'].value_counts().head(10))

# Step 4: Test is_remote vs. fraudulent
print("\nIs Remote vs. Fraudulent (Mean Fraudulent Rate):")
print(df.groupby('is_remote')['fraudulent'].mean())
contingency_table = pd.crosstab(df['is_remote'], df['fraudulent'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Test for is_remote vs. fraudulent: Chi2 = {chi2:.4f}, p-value = {p:.4f}, Significant = {p < 0.05}")


"""---------------------------------------------------------------------------------------------"""

# Imputation and column dropping
# Create has_salary_range feature
df['has_salary_range'] = df['salary_range'].notnull().astype(int)

# Impute missing values
text_cols = ['company_profile', 'requirements', 'benefits']
categorical_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'location', 'department']
for col in text_cols + categorical_cols:
    df[col] = df[col].fillna('Unknown')

# Drop columns
df = df.drop(columns=['in_balanced_dataset', 'department', 'salary_range'])

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Check new shape
print("\nNew Dataset Shape:", df.shape)


# Check columns
print("\nColumns After Processing:")
print(df.columns)

# Re-apply splitting and state cleaning
df[['country', 'state', 'city']] = df['location'].apply(split_location)
df['state'] = df['state'].apply(lambda x: x if x == 'Unknown' or (x.isalpha() and len(x) >= 2) else 'Unknown')

#  Drop is_remote
df = df.drop(columns=['is_remote'])

#  Re-check split location and unique values
print("\nSample of Split Location After Fix (First 10 Rows):")
print(df[['location', 'country', 'state', 'city']].head(10))
print("\nUnique States (Sample):")
print(df['state'].value_counts().head(10))
print("\nUnique Cities (Sample):")
print(df['city'].value_counts().head(10))

print("\nDropping 'location' column:")
df = df.drop(columns=['location'])
print("Shape after dropping location:", df.shape)


# Step 5: Verify imputation and column structure
print("\nMissing Values After Imputation:")
print(df.isnull().sum())
print("\nNew Dataset Shape:", df.shape)
print("\nColumns After Processing:")
print(df.columns)

"""----------------------------------------------------------------------------------"""
#Preprocess Text Columns

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to preprocess text (tokenize, remove stopwords, lemmatize)
def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Step 1: Clean text columns
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].apply(clean_text)

# Step 2: Preprocess text (tokenize, remove stopwords, lemmatize)
for col in text_cols:
    df[col] = df[col].apply(preprocess_text)

# Step 3: Combine text columns into a single column for TF-IDF
df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)

# Step 4: Apply TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=8000)
tfidf_features = tfidf.fit_transform(df['combined_text'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())

# Step 5: Drop original text columns and combined_text
df = df.drop(columns=text_cols + ['combined_text'])

# Step 7: Check new shape and sample TF-IDF columns
print("\nNew Dataset Shape After TF-IDF:", df.shape)
print("\nSample TF-IDF Columns (First 10):")
print(tfidf_df.columns[:10])
print("\nSample TF-IDF Data (First 5 Rows, First 10 Columns):")
print(tfidf_df.iloc[:5, :10])

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("TF-IDF Vectorizer saved successfully.")

"""------------------------------------------------------------------------------------"""

# Step 1: Verify the state of df (end of Step 5, after fixing fraudulent)
print("\nShape of df:", df.shape)
print("Columns in df:")
print(df.columns)
print("\nShape of df['fraudulent']:", df['fraudulent'].shape)
print("Type of df['fraudulent']:", type(df['fraudulent']))
print("First 5 rows of df['fraudulent']:\n", df['fraudulent'].head())

# Step 2: Identify duplicate columns
print("\nAny duplicate columns?", df.columns.duplicated().any())
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
print("Duplicate columns:", duplicate_cols)

# Step 5: Encode categorical columns
# Low-cardinality columns for one-hot encoding
low_cardinality_cols = ['employment_type', 'required_experience']
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoded = one_hot_encoder.fit_transform(df[low_cardinality_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(low_cardinality_cols))

# High-cardinality columns for target encoding
high_cardinality_cols = ['required_education', 'industry', 'function', 'country', 'state', 'city']
# Ensure all columns are treated as strings (categorical)
for col in high_cardinality_cols:
    df[col] = df[col].astype(str)

target_encoder = TargetEncoder()
target_encoded = target_encoder.fit_transform(df[high_cardinality_cols], df['fraudulent'])
target_encoded_df = pd.DataFrame(target_encoded, columns=high_cardinality_cols)

# Step 6: Drop original categorical columns and merge encoded ones
df = df.drop(columns=low_cardinality_cols + high_cardinality_cols)
df = pd.concat([df, one_hot_df, target_encoded_df], axis=1)

joblib.dump(one_hot_encoder, 'one_hot_encoder.pkl')
print("OneHotEncoder saved successfully.")

joblib.dump(target_encoder, 'target_encoder.pkl')
print("TargetEncoder saved successfully.")

# Step 7: Check new shape
print("\nNew Dataset Shape After Encoding:", df.shape)
print("\nSample Columns After Encoding (First 10):")
print(df.columns)

"""--------------------------------------------------------------------------------"""
# Step 7: Drop columns from tfidf_df that overlap with df to avoid duplicates
# Get the column names in df
df_columns = df.columns.tolist()
print("\nColumns in df:", df_columns)
print("Number of columns in df:", len(df_columns))

# Get the column names in tfidf_df that overlap with df
overlap_cols = [col for col in tfidf_df.columns if col in df_columns]
print("\nOverlapping columns in tfidf_df:", overlap_cols)

# Drop the overlapping columns from tfidf_df
tfidf_df_cleaned = tfidf_df.drop(columns=overlap_cols)
print("\nShape of tfidf_df after dropping overlapping columns:", tfidf_df_cleaned.shape)

# Ensure indices are aligned
df = df.reset_index(drop=True)
tfidf_df_cleaned = tfidf_df_cleaned.reset_index(drop=True)

# Concatenate df with the cleaned tfidf_df
df = pd.concat([df, tfidf_df_cleaned], axis=1)
print("\nShape after concatenating with tfidf_df:", df.shape)

# Verify no duplicates
print("\nAny duplicate columns after concatenation?", df.columns.duplicated().any())
if df.columns.duplicated().any():
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    print("Duplicate columns after concatenation:", duplicate_cols)
else:
    print("No duplicate columns after concatenation.")

df['telecommuting'].unique()

"""-------------------------------------------------------------------------------------"""
# Step 8: Chi-Square test for one-hot encoded columns
print("\nChi-Square Tests for One-Hot Encoded Features vs. Fraudulent:")
for col in one_hot_df.columns:
    contingency_table = pd.crosstab(df[col], df['fraudulent'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{col}: Chi2 = {chi2:.4f}, p-value = {p:.4f}, Significant = {p < 0.05}")

# Step 9: Chi-Square test for target-encoded columns (bin them first)
print("\nChi-Square Tests for Target-Encoded Features vs. Fraudulent:")
for col in high_cardinality_cols:
    # Bin the target-encoded values into quartiles
    df[f'{col}_binned'] = pd.qcut(df[col], q=4, duplicates='drop', labels=False)
    contingency_table = pd.crosstab(df[f'{col}_binned'], df['fraudulent'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{col}: Chi2 = {chi2:.4f}, p-value = {p:.4f}, Significant = {p < 0.05}")
    # Drop the binned column
    df = df.drop(columns=[f'{col}_binned'])               

# Step 10: Final shape and sample columns
print("\nFinal Dataset Shape:", df.shape)
print("\nSample Columns (First 10):")
print(df.columns[:10])

"""---------------------------------------------------------------------------"""

# Step 11: Compute Correlations with Target (fraudulent)
print("\nComputing Correlations with Target (fraudulent)...")

# Prepare features (X) and target (y)
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']

# Compute correlations (Spearman, as features include both continuous and binary)
correlations = X.corrwith(y, method='spearman')
# Sort by absolute correlation value
correlations_sorted = correlations.abs().sort_values(ascending=False)
print("Top 20 Features by Absolute Correlation with fraudulent:")
print(correlations_sorted.head(20))

#print the correlations with their signs (not just absolute values)
correlations_with_sign = correlations.sort_values(ascending=False)
print("\nTop 10 Positive Correlations with fraudulent:")
print(correlations_with_sign.head(10))
print("\nTop 10 Negative Correlations with fraudulent:")
print(correlations_with_sign.tail(10))

"""--------------------------------------------------------------------------------------"""

# Step 12: Feature Selection with Lasso and Mann-Whitney U Tests
print("\nPerforming Feature Selection with Lasso and Statistical Tests...")

# Prepare features (X) and target (y)
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']

# Standardize features (Lasso and statistical tests work better with scaled features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Apply Lasso (L1 regularization) for feature selection
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lasso.fit(X_scaled, y)

# Debug: Check the shape of lasso.coef_
print("Shape of lasso.coef_:", lasso.coef_.shape)

# For binary classification, lasso.coef_ should be a 1D array of shape (n_features,)
# If it's 2D with shape (1, n_features), we need to flatten it
if len(lasso.coef_.shape) == 2:
    coef = lasso.coef_.flatten()
else:
    coef = lasso.coef_

# Debug: Check the length of coef and compare with number of features
print("Length of coef:", len(coef))
print("Number of features in X:", X.shape[1])

# Get selected features (non-zero coefficients)
selected_mask = coef != 0
print("Number of non-zero coefficients:", np.sum(selected_mask))
selected_features = X.columns[selected_mask].tolist()
print(f"Number of features selected by Lasso: {len(selected_features)}")
print("Selected features (first 20):", selected_features[:20])

# Bar plot of top 20 Lasso coefficients
lasso_coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef}).sort_values(by='Coefficient', key=abs, ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=lasso_coef_df.head(20))
plt.title('Top 20 Features by Lasso Coefficient')
plt.xlabel('Lasso Coefficient')
plt.ylabel('Feature')
plt.show()

# Subset the DataFrame to selected features
X_selected = X[selected_features]
X_selected_scaled = X_scaled_df[selected_features]

# Perform Mann-Whitney U tests for selected features
print("\nPerforming Mann-Whitney U tests for selected features...")
significant_features = []
p_values = []
for col in selected_features:
    # Split feature values by class (fraudulent=0 vs fraudulent=1)
    group0 = X_selected[col][y == 0]
    group1 = X_selected[col][y == 1]
    
    # Perform Mann-Whitney U test (non-parametric, suitable for non-normal distributions)
    stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    if p < 0.05:
        significant_features.append(col)
    print(f"{col}: Mann-Whitney U p-value = {p:.4f}, Significant = {p < 0.05}")
    p_values.append(p)

print(f"\nNumber of significant features (Mann-Whitney U, p < 0.05): {len(significant_features)}")
print("Significant features (first 20):", significant_features[:20])


# Histogram of Mann-Whitney U test p-values
plt.figure(figsize=(8, 6))
plt.hist(p_values, bins=20, color='skyblue', edgecolor='black')
plt.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
plt.title('Distribution of Mann-Whitney U Test p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Final shape of selected features dataset
print("\nShape of X_selected:", X_selected.shape)
print("Sample Columns (First 10):")
print(X_selected.columns[:10])

"""-------------------------------------------------------------------------------------------------"""


# Step 13: Subset to Significant Features, Train-Test Split, and Handle Class Imbalance with SMOTENC
# Subset X_selected to significant features (from Mann-Whitney U tests)
X_significant = X_selected[significant_features]
print("\nShape of X_significant (after keeping only significant features):", X_significant.shape)
print("Sample Columns (First 10):")
print(X_significant.columns[:10])

with open('significant_features.pkl', 'wb') as f:
    pickle.dump(significant_features, f)
print("Significant features list saved successfully.")

# Define the potential binary columns (including telecommuting)
potential_binary_cols = [
    'telecommuting', 'has_company_logo', 'has_questions', 'company_profile_missing', 
    'requirements_missing', 'benefits_missing', 'employment_type_missing', 
    'required_experience_missing', 'required_education_missing', 'industry_missing', 
    'function_missing', 'location_missing', 'department_missing', 'salary_range_missing', 
    'has_salary_range'
]

# Add one-hot encoded columns (also binary: 0 or 1)
potential_binary_cols += [col for col in X_significant.columns if col.startswith('employment_type_') or col.startswith('required_experience_')]

# Check which of these binary columns are in X_significant
binary_cols = [col for col in potential_binary_cols if col in X_significant.columns]
print("\nBinary columns present in X_significant:")
print(binary_cols)
print("Number of binary columns:", len(binary_cols))

# Get the indices of binary columns in X_significant for SMOTENC
categorical_indices = [X_significant.columns.get_loc(col) for col in binary_cols]
print("\nCategorical indices for SMOTENC:", categorical_indices)

# Remove duplicates from binary_cols
binary_cols = list(dict.fromkeys(binary_cols))  # Preserve order while removing duplicates
print("\nBinary columns present in X_significant (after removing duplicates):")
print(binary_cols)
print("Number of binary columns (after removing duplicates):", len(binary_cols))

# Recalculate categorical indices for SMOTENC
categorical_indices = [X_significant.columns.get_loc(col) for col in binary_cols]
print("\nCategorical indices for SMOTENC (after removing duplicates):", categorical_indices)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_significant, y, test_size=0.2, random_state=42, stratify=y)
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Check class distribution in the training set
print("\nClass distribution in y_train (before SMOTENC):")
print(pd.Series(y_train).value_counts())

# Apply SMOTENC to the training set to handle class imbalance
smotenc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_smotenc, y_train_smotenc = smotenc.fit_resample(X_train, y_train)
print("\nShape of X_train after SMOTENC:", X_train_smotenc.shape)
print("Class distribution in y_train after SMOTENC:")
print(pd.Series(y_train_smotenc).value_counts())

# Verify that binary columns remain binary in the synthetic samples
print("\nChecking if binary columns remain binary after SMOTENC...")
for col in binary_cols[:5]:  # Check the first 5 binary columns
    unique_values = X_train_smotenc[col].unique()
    print(f"Unique values in {col} after SMOTENC:", unique_values)

"""--------------------------------------------------------------------"""

# Set a style for better-looking plots
plt.style.use('seaborn')

# Step 14: Train and Evaluate a Single Model with Visualizations and Threshold Adjustment

# Scale the data (since we have continuous features like TF-IDF and target-encoded features)
scaler = StandardScaler()
X_train_smotenc_scaled = scaler.fit_transform(X_train_smotenc)
X_test_scaled = scaler.transform(X_test)

# Plot class distribution before and after SMOTENC (run this only once)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
pd.Series(y_train).value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution Before SMOTENC')
plt.xlabel('Class (fraudulent)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
pd.Series(y_train_smotenc).value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution After SMOTENC')
plt.xlabel('Class (fraudulent)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Dictionary to store predictions and probabilities for McNemar's test
predictions = {}
probas = {}
roc_curves = {}
pr_curves = {}

# Define the model to train (change this variable to switch models)
model_name = "LightGBM" # Options: "Logistic Regression", "Random Forest", "XGBoost", "LightGBM"
model = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42)
}[model_name]

# Store metrics for all models
model_metrics = {}
# Function to train and evaluate a single model
def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Store predictions and probabilities for McNemar's test and curves
    predictions[model_name] = y_pred
    probas[model_name] = y_pred_proba
    
    # Evaluate with default threshold (0.5)
    print(f"\n{model_name} Classification Report (Default Threshold = 0.5):")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Metrics
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix (Default Threshold):")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
    
    # Plot Confusion Matrix as Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Fraudulent', 'Fraudulent'],
                yticklabels=['Non-Fraudulent', 'Fraudulent'])
    plt.title(f'Confusion Matrix: {model_name} (Default Threshold)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Precision-Recall AUC and Threshold Adjustment
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Find optimal threshold (maximizing F1-score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Add small epsilon to avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold (maximizing F1-score): {optimal_threshold:.4f}")
    
    # Precision-Recall trade-off plot
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision-Recall Trade-off: {model_name}')
    plt.legend()
    plt.show()

    # Predict with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    print(f"\n{model_name} Classification Report (Optimal Threshold = {optimal_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal))
    
    # Confusion Matrix with Optimal Threshold
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    print(f"\n{model_name} Confusion Matrix (Optimal Threshold):")
    print(cm_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
    
    # Plot Confusion Matrix with Optimal Threshold
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Fraudulent', 'Fraudulent'],
                yticklabels=['Non-Fraudulent', 'Fraudulent'])
    plt.title(f'Confusion Matrix: {model_name} (Optimal Threshold = {optimal_threshold:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Store metrics
    model_metrics[model_name] = {
        'Precision': precision_score(y_test, y_pred_optimal),
        'Recall': recall_score(y_test, y_pred_optimal),
        'F1-score': f1_score(y_test, y_pred_optimal)
    }
    
    # Store ROC and PR curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_curves[model_name] = (fpr, tpr, roc_auc)
    pr_curves[model_name] = (recall, precision, pr_auc)
    
    # Plot ROC Curve for the current model
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()
    plt.show()
    
    # Plot Precision-Recall Curve for the current model
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.show()
    
    # Feature Importance for Tree-Based Models
    if model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        # Get feature importance
        if model_name == "Random Forest":
            importances = model.feature_importances_
        elif model_name == "XGBoost":
            importances = model.feature_importances_
        elif model_name == "LightGBM":
            importances = model.feature_importances_ / sum(model.feature_importances_)  # Normalize for LightGBM
        
        # Create a DataFrame of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X_significant.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Feature Importances: {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

# Run the training and evaluation for the specified model
train_evaluate_model(model_name, model, X_train_smotenc_scaled, y_train_smotenc, X_test_scaled, y_test)

# Plot model comparison
metrics_df = pd.DataFrame(model_metrics).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison: Precision, Recall, F1-score')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.show()

# After running all models, run the McNemar's test to compare Logistic Regression and Random Forest
# This requires predictions from both models, so run this after evaluating both
def run_mcnemar_test():
    if "Logistic Regression" in predictions and "Random Forest" in predictions:
        print("\nPerforming McNemar's Test to Compare Logistic Regression vs. Random Forest...")
        contingency_table = pd.crosstab(predictions["Logistic Regression"], predictions["Random Forest"])
        # Create a 2x2 table for McNemar's test (correct vs. incorrect predictions)
        table = np.zeros((2, 2))
        table[0, 0] = np.sum((predictions["Logistic Regression"] == y_test) & (predictions["Random Forest"] == y_test))  # Both correct
        table[0, 1] = np.sum((predictions["Logistic Regression"] == y_test) & (predictions["Random Forest"] != y_test))  # LR correct, RF incorrect
        table[1, 0] = np.sum((predictions["Logistic Regression"] != y_test) & (predictions["Random Forest"] == y_test))  # LR incorrect, RF correct
        table[1, 1] = np.sum((predictions["Logistic Regression"] != y_test) & (predictions["Random Forest"] != y_test))  # Both incorrect
        print("Contingency Table for McNemar's Test:")
        print(table)
        chi2, p = chi2_contingency(table, correction=True)[:2]  # McNemar's test uses chi-square with correction
        print(f"McNemar's Test: Chi2 = {chi2:.4f}, p-value = {p:.4f}, Significant = {p < 0.05}")
    else:
        print("\nMcNemar's Test requires predictions from both Logistic Regression and Random Forest. Run both models first.")

run_mcnemar_test()

"""---------------------------------------------------------------------------------------------------------------"""

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'scale_pos_weight': [1, 5, 10]  # Adjust for class imbalance
}

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='f1', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_smotenc_scaled, y_train_smotenc)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)

# Train the best model on the full training set
best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train_smotenc_scaled, y_train_smotenc)

# Evaluate the tuned model
y_pred = best_xgb_model.predict(X_test_scaled)
y_pred_proba = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\nTuned XGBoost Classification Report (Default Threshold = 0.5):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nTuned XGBoost Confusion Matrix (Default Threshold):")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# ROC-AUC and Precision-Recall AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Find optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold (maximizing F1-score): {optimal_threshold:.4f}")

# Predict with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print(f"\nTuned XGBoost Classification Report (Optimal Threshold = {optimal_threshold:.4f}):")
print(classification_report(y_test, y_pred_optimal))

# Confusion Matrix with Optimal Threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
print("\nTuned XGBoost Confusion Matrix (Optimal Threshold):")
print(cm_optimal)
tn, fp, fn, tp = cm_optimal.ravel()
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

"""-------------------------------------------------------------------------------------------------"""

print("X_train Data Types:")
print(X_train.dtypes)
print("\nNumber of int64 (likely categorical) columns:", sum(X_train.dtypes == 'int64'))
print("Number of float64 (likely numerical) columns:", sum(X_train.dtypes == 'float64'))

# Define categorical feature indices (include both int32 and int64)
categorical_features_indices = [X_train.columns.get_loc(col) for col in X_train.select_dtypes(include=['int32', 'int64']).columns]

# Verify the number of categorical features
print("Number of categorical features (int32/int64):", len(categorical_features_indices))
print("Categorical features:", [X_train.columns[i] for i in categorical_features_indices])

# Apply SMOTENC with a 5:1 ratio
smotenc_1 = SMOTENC(categorical_features=categorical_features_indices, sampling_strategy=0.2, random_state=42)
X_train_smotenc_5to1, y_train_smotenc_5to1 = smotenc_1.fit_resample(X_train, y_train)

# Verify the new class distribution
print("Class Distribution After SMOTENC (5:1):")
print(pd.Series(y_train_smotenc_5to1).value_counts())

# Scale the new data
X_train_smotenc_5to1_scaled = scaler.fit_transform(X_train_smotenc_5to1)
X_test_scaled = scaler.transform(X_test)


# Retrain the XGBoost model
xgb_model_1 = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model_1.fit(X_train_smotenc_5to1_scaled, y_train_smotenc_5to1)

# Evaluate
y_pred = xgb_model_1.predict(X_test_scaled)
y_pred_proba = xgb_model_1.predict_proba(X_test_scaled)[:, 1]

print("\nXGBoost (5:1 SMOTENC) Classification Report (Default Threshold = 0.5):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nXGBoost (5:1 SMOTENC) Confusion Matrix (Default Threshold):")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Fraudulent', 'Fraudulent'],
            yticklabels=['Non-Fraudulent', 'Fraudulent'])
plt.title(f'Confusion Matrix: XGBoost with 5:1 ratio SMOTENC (Default Threshold)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC and Precision-Recall AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Precision-Recall trade-off plot
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.4f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Precision-Recall Trade-off: XGBoost with 5:1 Ratio SMOTENC')
plt.legend()
plt.show()

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold (maximizing F1-score): {optimal_threshold:.4f}")

y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print(f"\nXGBoost (5:1 SMOTENC) Classification Report (Optimal Threshold = {optimal_threshold:.4f}):")
print(classification_report(y_test, y_pred_optimal))

cm_optimal = confusion_matrix(y_test, y_pred_optimal)
print("\nXGBoost (5:1 SMOTENC) Confusion Matrix (Optimal Threshold):")
print(cm_optimal)
tn, fp, fn, tp = cm_optimal.ravel()
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# Plot Confusion Matrix with Optimal Threshold
plt.figure(figsize=(6, 4))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Fraudulent', 'Fraudulent'],
            yticklabels=['Non-Fraudulent', 'Fraudulent'])
plt.title(f'Confusion Matrix: XGBoost with 5:1 ratio SMOTENC (Optimal Threshold = {optimal_threshold:.4f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Store ROC and PR curve data
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_curves[xgb_model_1] = (fpr, tpr, roc_auc)
pr_curves[xgb_model_1] = (recall, precision, pr_auc)
    
    # Plot ROC Curve for the current model
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost 5:1 ratio SMOTENC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve: XGBoost 5:1 ratio SMOTENC')
plt.legend()
plt.show()
    
    # Plot Precision-Recall Curve for the current model
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'XGBoost 5:1 Ratio SMOTENC (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve: XGBoost 5:1 Ratio SMOTENC')
plt.legend()
plt.show()

"""-------------------------------------------------------------------------------------------------------"""
# Save the trained XGBoost model
joblib.dump(model, 'xgb_fraud_detection_model.pkl')
print("XGBoost model saved successfully as 'xgb_fraud_detection_model.pkl'.")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully as 'scaler.pkl'.")

# Save the optimal threshold
# Replace 0.2694 with the optimal threshold printed by train_evaluate_model if different
optimal_threshold = 0.2694  # Update this based on the output of train_evaluate_model
with open('optimal_threshold.txt', 'w') as f:
    f.write(str(optimal_threshold))
print("Optimal threshold saved successfully as 'optimal_threshold.txt'.")

"""-----------------------------------------"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the saved model
model = joblib.load('xgb_fraud_detection_model.pkl')

# Get feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances: XGBoost (1:1 SMOTENC)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Print top 10 features
print("Top 10 Features:")
print(feature_importance_df.head(10))
"""-----------------------------------------------------"""

import joblib
# Load the model, scaler, and threshold
model = joblib.load('xgb_fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read())
# Prepare new data (assuming new_data is a DataFrame with the same features)
new_data_scaled = scaler.transform(new_data)
new_pred_proba = model.predict_proba(new_data_scaled)[:, 1]
new_predictions = (new_pred_proba >= optimal_threshold).astype(int)

"""--------------------------------------------"""
