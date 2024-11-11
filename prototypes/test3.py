import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('agg')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_vars = request.form.get('num_of_vars')
        csv_file = request.files.get('csv_upload')
        ind_var_predict = request.form.get('ind_var_predict')

        if not csv_file:
            error_message = "Please upload a CSV file."
            return render_template('index.html', error_message=error_message)

        if ind_var_predict:
            try:
                ind_var_predict = [float(x) for x in ind_var_predict.split(',')]
                if len(ind_var_predict) != int(num_vars):
                    error_message = f"Invalid input. Please enter {num_vars} comma-separated numerical values."
                    return render_template('index.html', error_message=error_message)
            except ValueError:
                error_message = "Invalid input. Please enter comma-separated numerical values."
                return render_template('index.html', error_message=error_message)

        results = calculate_regression(csv_file, num_vars, ind_var_predict)
        return render_template('index2.html', results=results)
    return render_template('index2.html')

@app.route('/about')
def about():
    return render_template('about.html')

def calculate_regression(csv_file, num_vars, ind_var_predict=None):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :int(num_vars)]  # Independent variables columns based on user input
    y = df.iloc[:, -1]  # Last column is the dependent variable

    model = LinearRegression()
    model.fit(X, y)

    corr_coef = np.sqrt(model.score(X, y))
    r_squared = r2_score(y, model.predict(X))

    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coefficients):
        equation += f" + {coef:.2f}x{i+1}"

    corr_coef_desc = get_correlation_description(corr_coef)

    plot_data_list = []
    for i in range(int(num_vars)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(X.iloc[:, i], y, color='blue', label='Data Points')
        x_grid = np.linspace(min(X.iloc[:, i]), max(X.iloc[:, i]), 100)
        y_pred = model.predict(np.column_stack([x_grid] * int(num_vars)))
        ax.plot(x_grid, y_pred, color='red', label='Regression Line')
        ax.set_xlabel(f'Independent Variable {i+1}')
        ax.set_ylabel('Dependent Variable')
        ax.legend()

        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png')
        plot_data.seek(0)
        plot_data_base64 = base64.b64encode(plot_data.getvalue()).decode('utf-8')
        plot_data_list.append(plot_data_base64)

    if ind_var_predict:
        Y_pred = model.predict(np.array(ind_var_predict).reshape(1, -1))
    else:
        Y_pred = model.predict(X)

    results = {
        'correlation_coefficient': corr_coef,
        'correlation_coefficient_description': corr_coef_desc,
        'r_squared': r_squared,
        'equation': equation,
        'predicted_values': Y_pred.tolist(),
        'plot_data_list': plot_data_list,
        'resultss' : "See Results"
    }

    return results

def get_correlation_description(corr_coef):
    if corr_coef < 0.3:
        return "Weak correlation"
    elif corr_coef < 0.5:
        return "Moderate correlation"
    else:
        return "Strong correlation"

if __name__ == "__main__":
    app.run(debug=True)