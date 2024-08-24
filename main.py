from utils.data_loader import load_data
from models.imputation import vshd_imputation
from models.feature_selection import select_features
from models.shem_model import build_shem_model
from models.metrics import calculate_metrics, export_metrics_to_csv
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = load_data('data/dataset.csv')

# Handle missing values
data = vshd_imputation(data)

# Split data into features and target
X = data.drop('Precipitation', axis=1)
y = data['Precipitation']

# Feature selection
selected_features = select_features(X, y)
X = X[selected_features]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build and train SHEM model
model = build_shem_model(X_train, y_train)

# Generate synthetic predictions (replace with actual model predictions)
y_pred_rf = np.random.choice(['Yes', 'No'], size=len(y_test))
y_pred_svm = np.random.choice(['Yes', 'No'], size=len(y_test))
y_pred_dt = np.random.choice(['Yes', 'No'], size=len(y_test))

# Convert y_test to the same format
y_test = np.array(y_test)

# Calculate metrics
metrics = calculate_metrics(y_test, y_pred_rf, y_pred_svm, y_pred_dt)

# Export metrics to CSV
export_metrics_to_csv(metrics)

print("Model training complete and metrics exported.")
