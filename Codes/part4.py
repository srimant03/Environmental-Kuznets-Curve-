import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read the data
data = pd.read_csv('merged_data_with_sdp_and_gini_updated.csv')

#check for null values in gini index
print(data['gini index'].isnull().sum())
#drop null values of gini index
data = data.dropna(subset=['gini index'])

#regress the equation 
# GW = beta_0 + beta_1.Y + beta_2.Y**2  + delta_1.GINI + u

import statsmodels.api as sm

X = data[['sdp', 'gini index']]
X['sdp_sq'] = X['sdp']**2
#X['sdp_cube'] = X['sdp']**3
#X['sdp_gini_interaction'] = X['sdp'] * X['gini index']
print(X.head())
X = sm.add_constant(X)
y = data['nitrate']

model = sm.OLS(y, X).fit()

print(model.summary())

print("Coefficient of SDP: ", model.params['sdp'])
print("Coefficient of SDP squared: ", model.params['sdp_sq'])
#print("Coefficient of SDP cubed: ", model.params['sdp_cube'])
print("Coefficient of Gini Index: ", model.params['gini index'])
print("Intercept: ", model.params['const'])
print("R-squared: ", model.rsquared)
print("Adjusted R-squared: ", model.rsquared_adj)
print("Standard Error of SDP: ", model.bse['sdp'])
print("Standard Error of SDP squared: ", model.bse['sdp_sq'])
#print("Standard Error of SDP cubed: ", model.bse['sdp_cube'])
print("Standard Error of Gini Index: ", model.bse['gini index'])
print("Standard Error of Intercept: ", model.bse['const'])
print("T-value of SDP: ", model.tvalues['sdp'])
print("T-value of SDP squared: ", model.tvalues['sdp_sq'])
#print("T-value of SDP cubed: ", model.tvalues['sdp_cube'])
print("T-value of Gini Index: ", model.tvalues['gini index'])
print("T-value of Intercept: ", model.tvalues['const'])
print("P-value of SDP: ", model.pvalues['sdp'])
print("P-value of SDP squared: ", model.pvalues['sdp_sq'])
#print("P-value of SDP cubed: ", model.pvalues['sdp_cube'])
print("P-value of Gini Index: ", model.pvalues['gini index'])
print("P-value of Intercept: ", model.pvalues['const'])


plt.figure(figsize=(10, 6))
x_values = np.linspace(data['sdp'].min(), data['sdp'].max(), 100)
y_values = model.params['const'] + model.params['sdp'] * x_values + model.params['sdp_sq'] * (x_values ** 2)  + model.params['gini index'] * data['gini index'].mean()
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.scatter(data['sdp'], data['nitrate'], color='green', alpha=0.7, label='Actual Data', s=10)
plt.title('Regression Line', fontsize=16)
plt.xlabel('SDP', fontsize=14)
plt.ylabel('Nitrate', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

predicted_values = model.predict(X)
residuals = y - predicted_values
print("Residuals:", residuals)
print("Mean Squared Error:", np.mean(residuals ** 2))

# Plot residuals against predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predicted_values, residuals, color='purple', alpha=0.7, label='Residuals', s=10)
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

#plot histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, color='purple', alpha=0.7)
plt.axvline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Residuals Line')
plt.title('Histogram of Residuals', fontsize=16)
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#create plot of residuals vs sdps
plt.figure(figsize=(10, 6))
plt.scatter(data['sdp'], residuals, color='purple', alpha=0.7, label='Residuals', s=10)
plt.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Residuals Line')
plt.title('Residuals vs. SDP', fontsize=16)
plt.xlabel('SDP', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#plot residuals vs gini index
plt.figure(figsize=(10, 6))
plt.scatter(data['gini index'], residuals, color='purple', alpha=0.7, label='Residuals', s=10)
plt.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Residuals Line')
plt.title('Residuals vs. Gini Index', fontsize=16)
plt.xlabel('Gini Index', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()








'''
influence = model.get_influence()

# Get DFBETAS
dfbetas = influence.dfbetas

# Get DFFITS
dffits = influence.dffits[0]

# Get leverage values
leverage = influence.hat_matrix_diag

print("DFBETAS:")
print(dfbetas)
print("DFFITS:")
print(dffits)
print("Leverage:")
print(leverage)'''

