from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

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

    fig = plt.figure(figsize=(10, 6))
    if num_vars == '2':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, color='blue', label='Data Points')
        x1_grid, x2_grid = np.meshgrid(np.linspace(min(X.iloc[:, 0]), max(X.iloc[:, 0]), 100),
                                       np.linspace(min(X.iloc[:, 1]), max(X.iloc[:, 1]), 100))
        y_pred = model.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
        ax.plot_surface(x1_grid, x2_grid, y_pred.reshape(x1_grid.shape), alpha=0.5, color='red', label='Regression Plane')
        ax.set_xlabel('Independent Variable 1')
        ax.set_ylabel('Independent Variable 2')
        ax.set_zlabel('Dependent Variable')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X.iloc[:, 0], y, color='blue', label='Data Points')
        x1_grid = np.linspace(min(X.iloc[:, 0]), max(X.iloc[:, 0]), 100)
        y_pred = model.predict(np.array(x1_grid).reshape(-1, 1))
        ax.plot(x1_grid, y_pred, color='red', label='Regression Line')
        ax.set_xlabel('Independent Variable 1')
        ax.set_ylabel('Dependent Variable')

    ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10)
    # plot_data = io.BytesIO()
    # plt.savefig(plot_data, format='png')
    # plot_data.seek(0)

    plt.savefig('static/regression_plot.png') 

    Y_pred = model.predict(X)

    results = {
        'correlation_coefficient': corr_coef,
        'r_squared': r_squared,
        'equation': equation,
        # 'plot_data': plot_data.getvalue()
        'predicted_values': Y_pred.tolist(),
        'plot_path': 'static/regression_plot.png'
    }

    return results

if __name__ == "__main__":
    app.run(debug=True)