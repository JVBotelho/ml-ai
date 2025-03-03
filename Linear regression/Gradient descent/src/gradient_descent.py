import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from linear_regression import gradient_descent, compute_cost, plot_cost_history


def main():
    parser = argparse.ArgumentParser(description='Gradient Descent for Linear Regression')
    parser.add_argument('data_path', type=str, help='Path to CSV data file')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--standardize', action='store_true', help='Standardize features')

    args = parser.parse_args()

    try:
        # Load and preprocess data
        data = pd.read_csv(args.data_path)

        # Handle missing values and categorical data
        data = data.fillna(0)
        data = pd.get_dummies(data, columns=['ocean_proximity'], prefix='op')

        # Convert all data to numeric types
        data = data.astype(np.float64)

        # Separate features and target
        X = data.drop('median_house_value', axis=1).values
        y = data['median_house_value'].values.reshape(-1, 1)

        # Feature standardization
        if args.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Add intercept term after standardization
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize parameters
        theta = np.zeros((X.shape[1], 1))

        # Run gradient descent
        theta, cost_history = gradient_descent(
            X, y, theta, args.alpha, args.iterations
        )

        print(f"\nOptimized parameters:\n{theta.flatten()}")
        print(f"Final cost: {compute_cost(X, y, theta):.4f}")

        plot_cost_history(cost_history)

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'data' in locals():
            print("Data sample:\n", data.head())
            print("\nData types:\n", data.dtypes)


if __name__ == '__main__':
    main()