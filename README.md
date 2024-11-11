# SpendTrend - Grocery Expenditure Prediction

SpendTrend is a web application that helps users predict their grocery expenditure using linear regression analysis. The application analyzes the relationship between various household factors and grocery spending patterns to provide accurate expenditure predictions.

## Features

- Multiple variable analysis (1-3 independent variables)
- Interactive data visualization
- Correlation coefficient analysis
- R-squared value calculation
- Standard error estimation
- Regression equation generation
- CSV data upload functionality
- Responsive web design

## Technology Stack

- Frontend: HTML, CSS
- Backend: Python, Flask
- Data Analysis: NumPy, Pandas
- Machine Learning: scikit-learn
- Visualization: Matplotlib

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install flask pandas numpy sklearn matplotlib
```
3. Run the application:
```bash
python app.py
```

## Usage

1. **Select Variables**: Choose between 1-3 independent variables for your analysis
2. **Upload Data**: Prepare and upload a CSV file with your data
   - Place dependent variable (TOTAL GROCERY EXPENDITURE) in the rightmost column
   - Position independent variables to the left of the dependent variable
3. **Input Prediction Values**: Enter the values for your independent variables
4. **View Results**: Analyze the generated statistics and visualizations

## Recommended Variables

### Independent Variables:
- Total Household Income
- Total Number of Family Members
- Household Head Age

### Dependent Variable:
- Total Grocery Expenditure

## Data Format Requirements

1. CSV file format required
2. Dependent variable must be in the rightmost column
3. Independent variables should be arranged from left to right
4. Data should be numerical and properly formatted

## Output Information

The application provides:
- Correlation coefficient with description
- R-squared value
- Standard error of estimate
- Regression equation
- Predicted grocery expenditure
- Scatter plots with regression lines

## Team

- Mark Buduan - Back-End Developer
- Regine Keele - Front-End Developer
- Marc Nacpil - Front-End Developer

## Purpose

SpendTrend aims to help users:
- Better understand their grocery spending patterns
- Make informed budgeting decisions
- Optimize their grocery shopping experience
- Manage household finances more effectively

## Notes

- The application assumes a linear relationship between variables
- Accuracy depends on the quality and quantity of input data
- Predictions are estimates based on historical data patterns

## Contributing

For any suggestions or improvements, please contact the development team.
