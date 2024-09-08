import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

# Specify the path to your Excel file
file_path = 'fittlyfData (1).xlsx'

# Load only the desired sheet by name or by index
sheet_name = 'creditcard'  # Replace with your sheet name or index (e.g., 0 for the first sheet)

# Load the specific sheet into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Convert columns with object dtype to numeric, setting errors='coerce' to turn non-numeric values into NaNs
for col in ['V2', 'V7', 'V9', 'V24']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaNs in the specified columns
df = df.dropna(subset=['V2', 'V7', 'V9', 'V24'])

# Select numerical features for PCA
features = ['Time', 'Amount'] + [col for col in df.columns if col.startswith('V')]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

y = df['Class']

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Define the model
iso_forest = IsolationForest()
#(contamination=0.01, random_state=42)

# Fit the model
iso_forest.fit(pca_features,y)

# Predict anomalies
iso_forest_preds = iso_forest.predict(pca_features)
# Convert -1 (anomaly) and 1 (normal) to 0 and 1
iso_forest_preds = (iso_forest_preds == -1).astype(int)

# Evaluate
print("Isolation Forest Classification Report:")
print(classification_report(y, iso_forest_preds))
print("Isolation Forest ROC-AUC Score:", roc_auc_score(y, iso_forest_preds))

# Filter and print rows that are predicted as fraudulent
fraudulent_predictions = df[iso_forest_preds == 1]
print("Fraudulent Transactions Predicted by the Model:")
print(fraudulent_predictions)

# Title for the Streamlit app
st.title("Credit Card Fraud Detection with Isolation Forest and PCA")

# Upload file
uploaded_file = st.file_uploader("Upload your credit card transaction file", type=["xlsx"])

if uploaded_file is not None:
    # Load the file into a pandas DataFrame
    df_test = pd.read_excel(uploaded_file, sheet_name='creditcard_test')

    # Display uploaded data
    st.write("Data Sample:")
    st.dataframe(df_test.head())

    # Step 1: Prepare the test data
    test_features = ['Time', 'Amount'] + [col for col in df_test.columns if col.startswith('V')]

    # Step 2: Scale the test data
    scaler = StandardScaler()
    scaled_test_features = scaler.fit_transform(df_test[test_features])

    # Step 3: Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_test_features = pca.fit_transform(scaled_test_features)
    st.pca_test_features.sample(15)
    st.pca_test_features.max()
    st.pca_test_features.min()
    st.pca_test_features
    # Step 4: Apply Isolation Forest for anomaly detection
    iso_forest_test_preds = iso_forest.predict(pca_test_features)
    iso_forest_test_preds = (iso_forest_test_preds == -1).astype(int)

    # Step 5: Add prediction labels
    label_mapping = {0: 'Non-Fraudulent', 1: 'Fraudulent'}
    df_test['Prediction Label'] = pd.Series(iso_forest_test_preds).map(label_mapping)

    # Step 6: Fill NaN values if necessary
    df_test['Prediction Label'].fillna('Non-Fraudulent', inplace=True)

    # Display results
    st.write("Detection Results:")
    st.dataframe(df_test.head())

    # Step 7: Plot the data after PCA transformation
    st.write("Scatter Plot: PCA Components (Colored by Fraudulent/Non-Fraudulent)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pca_test_features[:, 0], y=pca_test_features[:, 1], 
                    hue=df_test['Prediction Label'], 
                    palette={'Non-Fraudulent': 'green', 'Fraudulent': 'red'}, ax=ax)
    ax.set_title("Scatter Plot: PCA Components 1 and 2")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # Step 8: Show fraudulent transactions predicted by the model
    st.write("Fraudulent Transactions Predicted by the Model:")
    fraudulent_test_predictions = df_test[df_test['Prediction Label'] == 'Fraudulent']
    st.dataframe(fraudulent_test_predictions)

else:
    st.write("Please upload a valid Excel file with the sheet name 'creditcard_test'.")
