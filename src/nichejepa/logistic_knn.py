import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

# Function to prepare features and labels from the DataFrame
def prepare_dataframes(df, feature_prefix='feature_', label_column='cluster_label', num_features=192):
    # Check if the specified label column exists in the DataFrame
    if label_column not in df.columns:
        print(f"{label_column} doesn't exist")
        return None, None

    # Create list of feature column names
    feature_columns = [f'{feature_prefix}{i}' for i in range(num_features)]
    
    # Extract feature values and ensure no NaNs in labels
    X = df[feature_columns].values
    le = LabelEncoder()  # Initialize LabelEncoder to encode labels
    y = df[label_column].dropna()  # Drop rows with NaN labels
    
    # Filter X to include only rows with non-NaN labels
    X = df.loc[y.index, feature_columns].values
    
    # Encode labels to integers
    y = le.fit_transform(y)

    return X, y

# Function to create PyTorch DataLoader objects from features and labels
def create_dataloaders(X, y, batch_size=32):
    # Create TensorDataset from features and labels
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    # Return DataLoader with batching and shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to train a model using data from a DataLoader
def train_model(model, train_loader):
    X_train = []
    y_train = []
    # Iterate over the DataLoader
    for X_batch, y_batch in train_loader:
        # Convert batches to numpy arrays and append to lists
        X_train.append(X_batch.numpy())
        y_train.append(y_batch.numpy())
    # Concatenate all batches into single arrays
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    # Train the model using the concatenated data
    model.fit(X_train, y_train)

# Function to evaluate a model's performance using data from a DataLoader
def evaluate_model(model, data_loader):
    X = []
    y_true = []
    # Iterate over the DataLoader
    for X_batch, y_batch in data_loader:
        # Convert batches to numpy arrays and append to lists
        X.append(X_batch.numpy())
        y_true.append(y_batch.numpy())
    # Concatenate all batches into single arrays
    X = np.concatenate(X, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    # Predict labels using the model
    y_pred = model.predict(X)
    # Calculate performance metrics
    return f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred)

# Function to run logistic regression classification
def run_logistic_regression(train_df, test_df, label_column, num_features=None):
    # Prepare training and testing data
    X_train, y_train = prepare_dataframes(train_df, label_column=label_column, num_features=num_features)
    X_test, y_test = prepare_dataframes(test_df, label_column=label_column, num_features=num_features)
    
    # Return None if data preparation failed
    if X_train is None or X_test is None:
        return None
    
    # Create DataLoaders for training and testing
    train_loader = create_dataloaders(X_train, y_train)
    test_loader = create_dataloaders(X_test, y_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    train_model(model, train_loader)

    # Evaluate model performance on training and test data
    train_f1, train_acc = evaluate_model(model, train_loader)
    test_f1, test_acc = evaluate_model(model, test_loader)

    # Print performance metrics
    print(f"Logistic Regression Train F1 Score: {train_f1}, Train Accuracy: {train_acc}")
    print(f"Logistic Regression Test F1 Score: {test_f1}, Test Accuracy: {test_acc}")

    return test_acc

# Function to run KNN classification
def run_knn_classification(train_df, test_df, label_column, num_features=None, n_neighbors=4):
    # Prepare training and testing data
    X_train, y_train = prepare_dataframes(train_df, label_column=label_column, num_features=num_features)
    X_test, y_test = prepare_dataframes(test_df, label_column=label_column, num_features=num_features)
    
    # Return None if data preparation failed
    if X_train is None or X_test is None:
        return None
    
    # Create DataLoaders for training and testing
    train_loader = create_dataloaders(X_train, y_train)
    test_loader = create_dataloaders(X_test, y_test)

    # Initialize and train the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    train_model(model, train_loader)

    # Evaluate model performance on training and test data
    train_f1, train_acc = evaluate_model(model, train_loader)
    test_f1, test_acc = evaluate_model(model, test_loader)

    # Print performance metrics
    print(f"KNN Classifier Train F1 Score: {train_f1}, Train Accuracy: {train_acc}")
    print(f"KNN Classifier Test F1 Score: {test_f1}, Test Accuracy: {test_acc}")

    return test_acc

# Function to run both logistic regression and KNN classification on the dataset
def logistic_and_knn(df, num_features=None):
    # Split the dataframe into training and testing sets based on 'split' column
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    # Logistic Regression
    test_acc_niche_logistic = run_logistic_regression(train_df, test_df, 'niche_type', num_features=num_features)
    test_acc_cell_logistic = run_logistic_regression(train_df, test_df, 'cell_type', num_features=num_features)

    # KNN Classification
    test_acc_niche_knn = run_knn_classification(train_df, test_df, 'niche_type', num_features=num_features)
    test_acc_cell_knn = run_knn_classification(train_df, test_df, 'cell_type', num_features=num_features)

    # Print final results
    print(f"Logistic Regression Niche Label Test Accuracy: {test_acc_niche_logistic}")
    print(f"Logistic Regression Cell Type Test Accuracy: {test_acc_cell_logistic}")
    print(f"KNN Classifier Niche Label Test Accuracy: {test_acc_niche_knn}")
    print(f"KNN Classifier Cell Type Test Accuracy: {test_acc_cell_knn}")

    # Return a dictionary of results
    return {
        "logistic_regression": {
            "niche_type": test_acc_niche_logistic,
            "cell_type": test_acc_cell_logistic
        },
        "knn_classifier": {
            "niche_type": test_acc_niche_knn,
            "cell_type": test_acc_cell_knn
        }
    }

