import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'winequality-red.csv')
    print(f"1. Loading dataset from: {csv_path}")
    wine_dataset = pd.read_csv(csv_path)

    print(f"   Dataset loaded successfully. Shape: {wine_dataset.shape} (rows, columns)")
    print(f"   Missing values: {wine_dataset.isnull().sum().sum()}")

    # Separate features and target
    X = wine_dataset.drop('quality', axis=1)
    # Binarize quality: 7 or higher is Good (1), otherwise Bad (0)
    Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

    print("\n2. Splitting data into 80% train and 20% test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    print("3. Training RandomForestClassifier model...")
    model = RandomForestClassifier(random_state=3)
    model.fit(X_train, Y_train)

    # Evaluate on training data
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(train_pred, Y_train)
    print(f"   Training Accuracy: {train_acc * 100:.2f}%")

    # Evaluate on test data
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(test_pred, Y_test)
    print(f"   Testing Accuracy:  {test_acc * 100:.2f}%")

    print("\n4. Running sample prediction test:")
    sample_features = {
        'fixed acidity': 7.5,
        'volatile acidity': 0.5,
        'citric acid': 0.36,
        'residual sugar': 6.1,
        'chlorides': 0.071,
        'free sulfur dioxide': 17.0,
        'total sulfur dioxide': 102.0,
        'density': 0.9978,
        'pH': 3.35,
        'sulphates': 0.8,
        'alcohol': 10.5
    }

    input_df = pd.DataFrame([sample_features])
    print("   Input features:")
    for k, v in sample_features.items():
        print(f"     - {k}: {v}")

    prediction = model.predict(input_df)
    result = "Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine"
    print(f"\n   Model Prediction: [{prediction[0]}] -> {result}")

    # Another sample test (High quality wine candidate)
    sample_good = {
        'fixed acidity': 8.5,
        'volatile acidity': 0.28,
        'citric acid': 0.56,
        'residual sugar': 1.8,
        'chlorides': 0.092,
        'free sulfur dioxide': 35.0,
        'total sulfur dioxide': 103.0,
        'density': 0.9969,
        'pH': 3.3,
        'sulphates': 0.75,
        'alcohol': 12.8
    }
    pred_good = model.predict(pd.DataFrame([sample_good]))
    result_good = "Good Quality Wine" if pred_good[0] == 1 else "Bad Quality Wine"
    print(f"\n   Second Sample (High alcohol, low volatile acidity):")
    print(f"   Model Prediction: [{pred_good[0]}] -> {result_good}")

if __name__ == '__main__':
    main()
