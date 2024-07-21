import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def prepare_dataframes(df, feature_prefix='feature_', label_column='cluster_label', num_features=192):
    if label_column not in df.columns:
        print(f"{label_column} doesn't exist")
        return None, None
    
    feature_columns = [f'{feature_prefix}{i}' for i in range(num_features)]
    X = df[feature_columns].values

    le = LabelEncoder()
    y = df[label_column].dropna()
    X = df.loc[y.index, feature_columns].values
    y = le.fit_transform(y)
    
    return X, y

def create_dataloaders(X, y, batch_size=32):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader):
    X_train, y_train = [], []
    for X_batch, y_batch in train_loader:
        X_train.extend(X_batch.numpy())
        y_train.extend(y_batch.numpy())
    model.fit(X_train, y_train)

def evaluate_model(model, data_loader):
    X, y_true = [], []
    for X_batch, y_batch in data_loader:
        X.extend(X_batch.numpy())
        y_true.extend(y_batch.numpy())
    y_pred = model.predict(X)
    return f1_score(y_true, y_pred, average='weighted')

def run_logistic_regression(train_df, test_df, label_column,num_features=None):
    X_train, y_train = prepare_dataframes(train_df, label_column=label_column, num_features=num_features)
    X_test, y_test = prepare_dataframes(test_df, label_column=label_column, num_features=num_features)
    if X_train is None or X_test is None:
      return None
    train_loader = create_dataloaders(X_train, y_train)
    test_loader = create_dataloaders(X_test, y_test)

    model = LogisticRegression(max_iter=400, solver='lbfgs')
    #model = LogisticRegression()
    train_model(model, train_loader)
    test_f1_weighted = evaluate_model(model, test_loader)

    return test_f1_weighted

def logistic_(df,num_features=None):
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    test_f1_niche = run_logistic_regression(train_df, test_df, 'niche_type',num_features=num_features)
    test_f1_cell = run_logistic_regression(train_df, test_df, 'cell_type',num_features=num_features)

    print(f"Niche Label Test Weighted F1 Score: {test_f1_niche}")
    print(f"Cell Type Test Weighted F1 Score: {test_f1_cell}")
    return test_f1_cell, test_f1_niche
