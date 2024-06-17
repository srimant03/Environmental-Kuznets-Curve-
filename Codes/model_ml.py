import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
from scipy.stats import t

# Load the data
data = pd.read_csv('merged_data_with_sdp_and_gini.csv')

# Define the independent variable (SDP) and the dependent variable (GWQ)
X = data[['sdp']]
y = data[['nitrate']]

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Calculate the residuals
residuals = y - predictions

# Calculate the mean squared error
mse = mean_squared_error(y, predictions)

# Calculate the R-squared value
r_squared = model.score(X, y)

# Adjusted R-squared
n = len(y)
p = X.shape[1]
adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))

# Calculate the standard errors of the coefficients
X_with_intercept = np.column_stack((np.ones_like(X), X))
X_inv = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
std_errors = np.sqrt(np.diagonal(X_inv * mse))

# Calculate t-values
t_values = model.coef_ / std_errors[1]

# Calculate p-values
p_values = (1 - t.cdf(np.abs(t_values), n - p)) * 2

# Print the metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adj_r_squared)
print("Standard Errors:", std_errors[1:])
print("T-values:", t_values)
print("P-values:", p_values)


#plot original data points
plt.scatter(X, y, color='green', label='Original Data',s=5)

# Plot residual distances
for i in range(len(X)):
    if i==0:
        plt.plot([X.iloc[i], X.iloc[i]], [y.iloc[i], predictions[i]], color='skyblue',linewidth=0.4,label='Residual Distance')
    else:
        plt.plot([X.iloc[i], X.iloc[i]], [y.iloc[i], predictions[i]], color='skyblue',linewidth=0.4)


#plot predicted data points
plt.scatter(X, predictions, color='red', label='Predicted Data',s=5)

# Add labels and title
plt.xlabel('SDP')
plt.ylabel('Nitrate')
plt.title('Linear Regression Analysis')
plt.legend()
plt.savefig('Linear_Reg_Analysis.png',dpi=300)
plt.show()

# Create the first plot: GWQ-Nitrate vs. SDP
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='green', alpha=0.7, label='GWQ-Nitrate',s=10)
plt.title('GWQ-Nitrate vs. SDP', fontsize=16)
plt.xlabel('SDP', fontsize=14)
plt.ylabel('GWQ-Nitrate', fontsize=14)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero GWQ-Nitrate Line')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('GWQ-Nitrate_vs_sdp.png', dpi=300)
plt.show()

# Create the second plot: Residuals vs. SDP
plt.figure(figsize=(10, 6))
plt.scatter(X, residuals, color='purple', alpha=0.7, label='Residuals-ûi,t',s=10)
plt.title('Residuals-ûi,t vs. SDP', fontsize=16)
plt.xlabel('SDP', fontsize=14)
plt.ylabel('Residuals-ûi,t', fontsize=14)
plt.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Residuals Line')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('residuals-ûi,t_vs_sdp.png', dpi=300)
plt.show()


# Plot a histogram of the residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black', linewidth=0.5)  # Adjust the number of bins and linewidth
plt.title('Histogram of Residuals', fontsize=16)
plt.xlabel('Residuals-ûi,t', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)  # Add gridlines with dashed lines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('residuals_histogram.png', dpi=300)
plt.show()

# Verify that the sum of the residuals equals zero
sum_residuals = np.sum(residuals)
max_residuals = np.max(residuals)
print("Sum of Residuals:", sum_residuals)
print("max of Residuals:", max_residuals)

#plot graph of residuals vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='purple', alpha=0.7, label='Residuals', s=10)
plt.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Residuals Line')
plt.title('Residuals vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()