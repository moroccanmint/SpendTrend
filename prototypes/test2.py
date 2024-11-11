import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_vars = request.form.get('num_of_vars')
        csv_file = request.files.get('csv_upload')
        ind_var1 = request.form.get('ind_one')
        ind_var2 = request.form.get('ind_two')
        ind_var3 = request.form.get('ind_three')

        results = calculate_regression(num_vars, csv_file, ind_var1, ind_var2, ind_var3)
        return render_template('index.html', results=results)
    return render_template('index.html')

def calculate_regression(num_vars, csv_file, ind_var1, ind_var2=None, ind_var3=None):
    if csv_file:
        df = pd.read_csv(csv_file)
    else:
        X1 = [float(x) for x in ind_var1.split(',')]
        if num_vars == '2':
            X2 = [float(x) for x in ind_var2.split(',')]
            data = {'Independent Variable 1': X1, 'Independent Variable 2': X2}
        elif num_vars == '3':
            X2 = [float(x) for x in ind_var2.split(',')]
            X3 = [float(x) for x in ind_var3.split(',')]
            data = {'Independent Variable 1': X1, 'Independent Variable 2': X2, 'Independent Variable 3': X3}
        else:
            data = {'Independent Variable 1': X1}
        Y = [float(y) for y in request.form.get('dep_var').split(',')]
        data['Dependent Variable'] = Y
        df = pd.DataFrame(data)

    X = df.drop('Dependent Variable', axis=1)
    y = df['Dependent Variable']

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

    Y_pred = model.predict(X)

    results = {
        'correlation_coefficient': corr_coef,
        'correlation_coefficient_description': corr_coef_desc,
        'r_squared': r_squared,
        'equation': equation,
        'predicted_values': Y_pred.tolist(),
        'plot_data_list': plot_data_list
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