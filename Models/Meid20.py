# Importing required libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = 'datapathforyourtraining dataset'  # Replace with the correct path to your dataset
dataset = pd.read_pickle(data_path)

# Preprocessing the features
# Convert 'deviceName' to categorical and encode as integers
dataset['deviceName'] = dataset['deviceName'].astype('category')
dataset['device_cat'] = dataset['deviceName'].cat.codes

# Select Meid20 features and target variable
features = [
    'src_port', 'dst_port', 'protocol', 'dst2src_packets', 'dst2src_bytes',
    'src2dst_syn_packets', 'src2dst_cwr_packets', 'src2dst_ece_packets', 
    'src2dst_urg_packets', 'src2dst_ack_packets', 'src2dst_psh_packets', 
    'src2dst_rst_packets', 'src2dst_fin_packets', 'dst2src_syn_packets', 
    'dst2src_cwr_packets', 'dst2src_ece_packets', 'dst2src_urg_packets', 
    'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets', 
    'dst2src_fin_packets'
]
target = 'device_cat'

# Create a new DataFrame for training and testing
data = dataset[features + [target]]
data = data.set_index('start_time')  # Ensure 'start_time' is the index

# Split the data into features (X) and target (y)
X = data.drop(columns=[target])
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define and configure the LGBM model
model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=30,
    n_estimators=30,
    random_state=42
)

# Train the model using GPU if available
with tf.device('/GPU:0'):
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test), (X_train, y_train)],
        eval_metric='logloss',
        verbose=False
    )

# load the testing dataset usecase
data_test_path = 'datapathforyourtestingcasedataset'  # Replace with the correct path to your dataset
testset = pd.read_pickle(data_test_path)

X_test = testset.drop(columns=[target])
y_test = testset[target]

# Evaluate the model using weighted ROC-AUC
y_pred_proba = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

print(f"Weighted ROC-AUC Score: {roc_auc:.4f}")
