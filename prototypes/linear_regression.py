import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_regression():
    # Get values from the form
    num_vars = request.form.get('num_of_vars')
    ind_var1 = request.form.get('ind_one').split(',')
    ind_var2 = request.form.get('ind_two').split(',')
    ind_var3 = request.form.get('ind_three').split(',')
    csv_file = request.files.get('csv_upload')

    # Check if file uploaded (might be empty)
    if csv_file:
        df = pd.read_csv(csv_file)
        X1 = df['Independent Variable 1']
        X2 = df['Independent Variable 2']
        Y = df['Dependent Variable']
    else:
        return {}  # Return an empty dictionary if no CSV file is provided

    # Convert the input strings to lists of floats
    pred_X1 = [float(x) for x in ind_var1]
    pred_X2 = [float(x) for x in ind_var2] if ind_var2 else None
    pred_X3 = [float(x) for x in ind_var3] if ind_var3 else None

    # Create a DataFrame
    if num_vars == '2':
        data = {'Independent Variable 1': X1, 'Independent Variable 2': X2, 'Dependent Variable': Y}
    else:
        data = {'Independent Variable 1': X1, 'Dependent Variable': Y}
    df = pd.DataFrame(data)

    # Perform multiple regression
    if num_vars == '2':
        X = df[['Independent Variable 1', 'Independent Variable 2']]
    else:
        X = df[['Independent Variable 1']]
    y = df['Dependent Variable']
    model = LinearRegression()
    model.fit(X, y)

    # Calculate the correlation coefficient
    corr_coef = np.sqrt(model.score(X, y))
    print("Correlation Coefficient:", corr_coef)

    # Calculate the R-squared value
    r_squared = r2_score(y, model.predict(X))
    print("R-squared:", r_squared)

    # Plot the regression line
    fig = plt.figure(figsize=(10, 6))
    if num_vars == '2':
        ax = fig.add_subplot(111, projection='3d')
        # Plot the data points
        ax.scatter(X1, X2, Y, color='blue', label='Data Points')
        # Plot the regression plane
        x1_grid, x2_grid = np.meshgrid(np.linspace(min(X1), max(X1), 100), np.linspace(min(X2), max(X2), 100))
        y_pred = model.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
        ax.plot_surface(x1_grid, x2_grid, y_pred.reshape(x1_grid.shape), alpha=0.5, color='red', label='Regression Plane')
        ax.set_xlabel('Independent Variable 1')
        ax.set_ylabel('Independent Variable 2')
    else:
        ax = fig.add_subplot(111)
        # Plot the data points
        ax.scatter(X1, Y, color='blue', label='Data Points')
        # Plot the regression line
        x1_grid = np.linspace(min(X1), max(X1), 100)
        y_pred = model.predict(np.array(x1_grid).reshape(-1, 1))
        ax.plot(x1_grid, y_pred, color='red', label='Regression Line')
        ax.set_xlabel('Independent Variable 1')

    ax.set_zlabel('Dependent Variable')

    # Display the equation of the regression plane
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coefficients):
        equation += f" + {coef:.2f}x{i+1}"
    ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10)

    plt.savefig('static/regression_plot.png')  # Save the plot as an image

    # Predict the values of the dependent variable (Y) for each combination of X1, X2, and X3
    Y_pred = []
    if num_vars == '2':
        for x1, x2 in zip(pred_X1, pred_X2):
            y_pred = intercept + coefficients[0] * x1 + coefficients[1] * x2
            Y_pred.append(y_pred)
    else:
        for x1 in pred_X1:
            y_pred = intercept + coefficients[0] * x1
            Y_pred.append(y_pred)

    print("Predicted Y values:", Y_pred)

    # Create a dictionary to store the results
    results = {
        'correlation_coefficient': corr_coef,
        'r_squared': r_squared,
        'equation': equation,
        'predicted_values': Y_pred,
        'plot_path': 'static/regression_plot.png'
    }

    return results