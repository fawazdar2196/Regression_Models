import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Sample data points 
x = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(x, y)
x_new = np.linspace(0, 5, 100).reshape(-1, 1)
y_linear_pred = linear_model.predict(x_new)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)
x_new_poly = poly_features.transform(x_new)
y_poly_pred = poly_model.predict(x_new_poly)

# Polynomial Regression (degree 3)
poly_features3 = PolynomialFeatures(degree=3)
x_poly3 = poly_features3.fit_transform(x)
poly_model3 = LinearRegression()
poly_model3.fit(x_poly3, y)
x_new_poly3 = poly_features3.transform(x_new)
y_poly_pred3 = poly_model3.predict(x_new_poly3)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x, y)
y_ridge_pred = ridge_model.predict(x_new)

# Plotting the results
plt.figure(figsize=(14, 10))

# Plot Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_new, y_linear_pred, color='blue', label='Linear regression')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Linear Regression', fontsize=14)
plt.legend()
plt.grid(True)

# Plot Polynomial Regression (degree 2)
plt.subplot(2, 2, 2)
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_new, y_poly_pred, color='green', label='Polynomial regression (degree 2)')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Polynomial Regression (Degree 2)', fontsize=14)
plt.legend()
plt.grid(True)

# Plot Polynomial Regression (degree 3)
plt.subplot(2, 2, 3)
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_new, y_poly_pred3, color='purple', label='Polynomial regression (degree 3)')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Polynomial Regression (Degree 3)', fontsize=14)
plt.legend()
plt.grid(True)

# Plot Ridge Regression
plt.subplot(2, 2, 4)
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_new, y_ridge_pred, color='orange', label='Ridge regression')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Ridge Regression', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print performance metrics
def print_metrics(model, x, y, x_new, y_pred):
    y_train_pred = model.predict(x)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

print("Linear Regression Metrics:")
print_metrics(linear_model, x, y, x_new, y_linear_pred)

print("\nPolynomial Regression (Degree 2) Metrics:")
print_metrics(poly_model, x_poly, y, x_new_poly, y_poly_pred)

print("\nPolynomial Regression (Degree 3) Metrics:")
print_metrics(poly_model3, x_poly3, y, x_new_poly3, y_poly_pred3)

print("\nRidge Regression Metrics:")
print_metrics(ridge_model, x, y, x_new, y_ridge_pred)

# Discussion of results
print("\nDiscussion:")
print("The script includes multiple regression models: Linear Regression, Polynomial Regression (degrees 2 and 3), and Ridge Regression.")
print("Each model was fitted to the dataset, and the regression lines were plotted alongside the original data points.")
print("Performance metrics (Mean Squared Error and R^2 Score) were calculated to evaluate the models.")
print("Linear Regression provides a simple linear fit, while Polynomial Regression (degrees 2 and 3) captures more complex relationships.")
print("Ridge Regression adds a regularization term to Linear Regression to prevent overfitting.")
