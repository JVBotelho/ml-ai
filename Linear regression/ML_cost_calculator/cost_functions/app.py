import numpy as np
from cost_functions.losses import (
    mean_squared_error,
    mean_absolute_error,
    binary_cross_entropy,
    categorical_cross_entropy
)


def main():
    print("ML Cost Function Calculator\n")

    # Get cost function type
    cost_type = input(
        "Choose cost function:\n1. MSE\n2. MAE\n3. Binary Cross-Entropy\n4. Categorical Cross-Entropy\n> ")

    # Get array inputs
    y_true = _parse_array_input("Enter true values (comma-separated): ")
    y_pred = _parse_array_input("Enter predicted values (comma-separated): ")

    try:
        if cost_type == '1':
            result = mean_squared_error(y_true, y_pred)
        elif cost_type == '2':
            result = mean_absolute_error(y_true, y_pred)
        elif cost_type == '3':
            result = binary_cross_entropy(y_true, y_pred)
        elif cost_type == '4':
            result = categorical_cross_entropy(y_true, y_pred)
        else:
            raise ValueError("Invalid choice")

        print(f"\nResult: {result:.4f}")
    except Exception as e:
        print(f"\nError: {str(e)}")


def _parse_array_input(prompt):
    while True:
        try:
            input_str = input(prompt)
            # Split rows by ; and elements by ,
            rows = [row.split(',') for row in input_str.split(';')]
            return np.array([[float(x.strip()) for x in row] for row in rows])
        except ValueError:
            print("Invalid input. Use format: 1.1,2.2;3.3,4.4")


if __name__ == "__main__":
    main()