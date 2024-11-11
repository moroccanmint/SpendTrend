import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('agg')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form_submitted = False
    if request.method == 'POST':
        num_vars = request.form.get('num_of_vars')
        csv_file = request.files.get('csv_upload')
        ind_var_predict = request.form.get('ind_var_predict')

        if not csv_file:
            error_message = "Please upload a CSV file."
            return render_template('home.html', error_message=error_message)

        if ind_var_predict:
            try:
                ind_var_predict = [float(x) for x in ind_var_predict.split(',')]
                if len(ind_var_predict) != int(num_vars):
                    error_message = f"Invalid input. Please enter {num_vars} comma-separated numerical values."
                    return render_template('home.html', error_message=error_message)
            except ValueError:
                error_message = "Invalid input. Please enter comma-separated numerical values."
                return render_template('home.html', error_message=error_message)

        results = calculate_regression(csv_file, num_vars, ind_var_predict)
        form_submitted = True
        return render_template('home.html', results=results)
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

def calculate_regression(csv_file, num_vars, ind_var_predict=None):
    df = pd.read_csv(csv_file)
    columns = df.columns

    X = df.iloc[:, :int(num_vars)]  # Independent variables columns based on user input
    y = df.iloc[:, -1]  # Last column is the dependent variable

    # Get the column names for the independent variables
    ind_var_names = columns[:int(num_vars)]

    # Get the column name for the dependent variable
    dep_var_name = columns[-1]

    model = LinearRegression()
    model.fit(X, y)
    corr_coef = np.sqrt(model.score(X, y))
    r_squared = r2_score(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))
    coefficients = model.coef_
    intercept = model.intercept_

    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coefficients):
        equation += f" + {coef:.2f}*{ind_var_names[i]}"

    corr_coef_desc = get_correlation_description(corr_coef)

    plot_data_list = []
    for i in range(int(num_vars)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(X.iloc[:, i], y, color='blue', label='Data Points')
        x_grid = np.linspace(min(X.iloc[:, i]), max(X.iloc[:, i]), 100)
        # Use the actual values of the independent variables for generating predictions
        y_pred = model.predict(X)
        # y_pred = model.predict(np.column_stack([x_grid] * int(num_vars)))
        # ax.plot(x_grid, y_pred, color='red', label='Regression Line')
        ax.plot(X.iloc[:, i], y_pred, color='red', label='Regression Line')
        ax.set_xlabel(ind_var_names[i])
        ax.set_ylabel(dep_var_name)
        ax.set_title(f'{dep_var_name} vs {ind_var_names[i]}')

        # # Set y-axis limits
        # ax.set_ylim(bottom=min(y), top=max(y))
    
        ax.legend()
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png')
        plot_data.seek(0)
        plot_data_base64 = base64.b64encode(plot_data.getvalue()).decode('utf-8')
        plot_data_list.append(plot_data_base64)

    if ind_var_predict:
        Y_pred = np.round(model.predict(np.array(ind_var_predict).reshape(1, -1)), 2)
    else:
        Y_pred = np.round(model.predict(X), 4)

    results = {
        'correlation_coefficient': round(corr_coef, 4),
        'correlation_coefficient_description': corr_coef_desc,
        'r_squared': round(r_squared, 4),
        'standard_error_estimate': round(np.sqrt(mse), 4),
        'equation': equation,
        'predicted_values': Y_pred.tolist(),
        'plot_data_list': plot_data_list,
        'resultss': "See Results"
    }

    return results

def get_correlation_description(corr_coef):
    if corr_coef < 0.1:
        return "Negligible correlation"
    elif corr_coef < 0.3:
        return "Weak correlation"
    elif corr_coef < 0.5:
        return "Moderate correlation"
    elif corr_coef < 0.7:
        return "Strong correlation"
    elif corr_coef < 0.9:
        return "Very strong correlation"
    else:
        return "Extremely strong correlation"

if __name__ == "__main__":
    app.run(debug=True)