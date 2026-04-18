
# for data manipulation
import pandas as pd
import sklearn
import os

# for splitting
from sklearn.model_selection import train_test_split

# for encoding categorical variables
from sklearn.preprocessing import LabelEncoder

# for hugging face upload
from huggingface_hub import HfApi

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path (update if needed)
DATASET_PATH = "hf://datasets/sharath96yp/Tourism-Prediction/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Target column
target_col = 'ProdTaken'

# Separate categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Initialize LabelEncoder dictionary
label_encoders = {}

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # handle missing safely
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test split completed and files saved.")

# Upload to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sharath96yp/Tourism-Prediction",
        repo_type="dataset",
    )

print("Files uploaded successfully.")
