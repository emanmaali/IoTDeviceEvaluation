import numpy as np
import pandas as pd
import pickle
import os 
import warnings
import logging
from keras import backend as K
import gc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,LSTM, Dense , RepeatVector, TimeDistributed, Dropout
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler  # MinMaxScaler
from memory_profiler import profile

warnings.filterwarnings("ignore", category=DeprecationWarning) #disable Deprecation Warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

value_to_fill = {'TCP RTO': 0, 'IP TTL': 0, 'IP DF': 0, 'IP TOS':0,'TCP WIN':0}

def train_lstm_model(X_train, y_train, n_classes, num_epochs,n_timesteps,n_features):
    logging.info('Train an yang LSTM model for multiclass..')
    """
    Train an LSTM model for multiclass classification.

    Args:
    - X_train (numpy.ndarray): Train input data.
    - y_train (numpy.ndarray): Train output labels.
    - num_classes (int): Number of classes.
    - num_epochs (int): Number of training epochs.

    Returns:
    - model (tensorflow.keras.Model): Trained LSTM model.
    """
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(n_timesteps,n_features)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(n_classes, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall()])
    lstm_model.summary() 
    try: 
       logging.info('trying to fit the model')
       lstm_model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)
       return lstm_model
    except Exception as e:
        print(f"An error occurred with traingin lstm model: {e}")


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def test_lstm_model(test_paths,model,dataset_name,selected_features,label_column): #model,test_paths
    results_dict = {}
    print(dataset_name)
    with open(test_paths, 'r') as f:
        for line in f:
            path, start_date, end_date = line.strip().split(',')
            try:
                logging.info(f'testing {path}..')
                # Call preprocess_data function
                df_lst_test, n_classes = preprocess_data(path,dataset_name, selected_features)
                # Check if df_lst_test is not empty
                if not df_lst_test.empty:
                    test_features, test_labels = split_data_by_date(df_lst_test,start_date, end_date, label_column)
                    X_test, y_test, n_timesteps, n_features, n_outputs = pileline_data(test_features, test_labels, 64, n_classes)
                    print(path,X_test.shape, y_test.shape, n_timesteps, n_features, n_outputs)
                    try : 
                       # Predict probabilities using the model
                       y_pred_prob = model.predict(X_test)
                    except Exception as e:
                        print(f"An error occurred with model.predict for{path}: {e}")
                    # Calculate TPR and FPR using ROC curve
                    try : 
                        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                        # Calculate precision-recall curve for each class
                        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_pred_prob.ravel())
                        print(precision, recall)
                        # Calculate AUC-PR for each class
                        aucpr = auc(recall, precision)
                        # Average AUC-PR across all classes
                        avg_aucpr = np.mean(aucpr)
                        # Store evaluation metrics in results dictionary
                        file_name = path.split('/')[-1]  # Extract file name from path
                        results_dict[file_name] = {'AUC-PR': avg_aucpr}
                        K.clear_session()
                        gc.collect()
                    except Exception as e:
                        print(f"An error occurred with calcualting TPR, FPR and AUCpr for{path}: {e}")    
            except Exception as e:
                    print(f"An error occurred with {path}: {e}")

    return results_dict


def save_results(results_dict, output_file):
    print(results_dict)
    """
    Save evaluation results to a text file.

    Args:
    - results_dict (dict): Dictionary containing evaluation metrics for each test dataset.
    - output_file (str): Path to the output text file.
    """
    with open(output_file, 'a') as f:
        for file_name, metrics in results_dict.items():
            f.write(f"File: {file_name}\n")
            # f.write(f"TPR: {metrics['TPR']}\n")
            # f.write(f"FPR: {metrics['FPR']}\n")
            f.write(f"AUC-PR: {metrics['AUC-PR']}\n\n")


def preprocess_dataframe(df):
    logging.info("converting column data types and adding a 'start_time' column...")
    """
    Preprocesses a DataFrame by converting column data types and adding a 'start_time' column.
    """
    # Define column data type conversions
    column_type_conversions = {
        'src': 'string',
        'dst': 'string',
        'proto': 'float64',
        'srcport': 'float64',
        'dstport': 'float64',
        'IP TTL': 'float64',
        'IP DF': 'float64',
        'TCP RTO':'float64', 
        'IP TOS':'float64',
        'TCP WIN':'float64',
        'datasetName': 'string',
        'device': 'string',
    }

    # Convert column data types
    df = df.astype(column_type_conversions)
    # Convert 'timestamp' column to datetime and assign to 'start_time'
    df['start_time'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def replace_missing_values(df):
    """
    Replace missing values (NaN or NA) in a DataFrame with suitable values based on column type.
    """
    logging.info("Replace missing values (NaN or NA) in a DataFrame with suitable values based on column type...")
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            # For string columns, replace missing values with an empty string
            df[column].fillna('', inplace=True)
        elif pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns, replace non-numeric values with NaN
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # Fill NaN with the mean of the column
            df[column].fillna(df[column].mean(), inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            # For datetime columns, replace missing values with NaT (Not a Time) object
            df[column].fillna(pd.NaT, inplace=True)
        else:
            # For other column types, replace missing values with None
            df[column].fillna(0, inplace=True)
    return df

def preprocess_data(file_path, dataset_name, features):
    logging.info("reading data..")
    new_test_active = pd.read_pickle(file_path)
    new_test_active.loc[new_test_active['datasetName'] == dataset_name]
    new_test_active = preprocess_dataframe(new_test_active)
    new_test_active =replace_missing_values(new_test_active)
    new_test_active = new_test_active.fillna(value=value_to_fill)
    print(new_test_active.isnull().values.any())   
    new_test_active = new_test_active.sort_index()
    new_test_active.reset_index(drop=True, inplace=True)
    new_test_active = new_test_active.set_index('start_time')
    new_test_active['device'] = new_test_active['device'].astype('category')
    new_test_active['device_cat'] = new_test_active['device'].cat.codes
    df_lst_new = new_test_active[features]
    return df_lst_new, df_lst_new['device_cat'].nunique()

# Extract training and test data based on dates
def split_data_by_date(df, start_date_train, end_date_train,label_column):
    logging.info('Extract data based on dates..')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    train_features = df.loc[start_date_train:end_date_train]
    train_labels = df.loc[start_date_train:end_date_train,label_column]
    train_features = train_features.drop(columns=label_column)
    return train_features, train_labels

def pileline_data(train_features, train_labels, T, n_classes):
    logging.info('Scale the features...')
    # Scale the features
    scaler = StandardScaler()
    scaled_train_features = pd.DataFrame(scaler.fit_transform(train_features.values),
                                         index=train_features.index,
                                         columns=train_features.columns)
    # Prepare the training data
    X_train, y_train = [], []
    for i in range(train_labels.shape[0] - (T-1)):
        X_train.append(scaled_train_features.iloc[i:i+T].values)
        y_train.append(train_labels.iloc[i + (T-1)])
    logging.info('after for loop in Scale the features...')
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)
    y_train = to_categorical(y_train, num_classes=n_classes)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    return X_train, y_train, n_timesteps, n_features, n_outputs

def read_test_paths(file_path):
    logging.info(f'reading test file..{file_path}')
    test_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            path, start_date, end_date = line.strip().split(',')
            test_paths.append((path.strip(), start_date.strip(), end_date.strip()))
    return test_paths

def read_train_file(file_path):
    logging.info('reading train file..')
    with open(file_path, 'r') as file:
        line = file.readline().strip()  # Assuming only one line is read
        parts = line.split(',')
        if len(parts) == 7:
            paths = parts[0].strip()
            dataset_names = parts[1].strip()
            start_dates = parts[2].strip()
            end_dates = parts[3].strip()
            epochs = int(parts[4].strip())
            test_paths = parts[5].strip()
            results_paths = parts[6].strip()
    return paths, dataset_names, start_dates, end_dates, epochs, test_paths, results_paths


def main():
    label_column = "device_cat"
    parsed_data = read_train_file('trainingdatasetpath')
    train_path, dataset_name, train_start_date, train_end_date, num_epochs, test_path, results_path = parsed_data
    
    selected_features = ['proto', 'srcport', 'dstport', 
                         'IP TTL', 'IP DF', 'TCP WIN', 'device_cat']

    # Generate training data
    df_lst_train, n_classes = preprocess_data(train_path, dataset_name, selected_features)
    train_features, train_labels = split_data_by_date(df_lst_train,train_start_date, train_end_date,label_column)
    X_train, y_train, n_timesteps, n_features, n_outputs = pileline_data(train_features, train_labels,64, n_classes)
    print(X_train.shape,y_train.shape,n_timesteps, n_features, n_outputs)
    # Train the model
    # After each iteration:
    with tf.device('/GPU:0'):
        model = train_lstm_model(X_train, y_train, n_classes, num_epochs,n_timesteps,n_features)
        # Test the model
        results_dict = test_lstm_model(test_path,model,dataset_name, selected_features,label_column) 
        save_results(results_dict, results_path)


if __name__ == "__main__":
    main()
