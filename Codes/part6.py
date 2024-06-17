import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read the data
data = pd.read_csv('merged_data_with_sdp_and_gini_updated.csv')

#regress the equation 
# GW = beta_0 + beta_1.Y + beta_2.Y**2 + beta_3.Y**3 + delta_1.GINI + u

print(data['gini index'].isnull().sum())

data = data.dropna(subset=['gini index'])

years = data['year'].unique()

import statsmodels.api as sm

sdp=[]
sdp_sq=[]
r_sq = []
for year in years:
    print("Year: ", year)
    data_year = data[data['year'] == year]
    X = data_year[['sdp', 'gini index']]
    X['sdp_sq'] = X['sdp']**2
    #X['sdp_gini_interaction'] = X['sdp'] * X['gini index']
    X = sm.add_constant(X)
    y = data_year['nitrate']

    model = sm.OLS(y, X).fit()
    predicted_values = model.predict(X)

    print("Coefficient of SDP: ", model.params['sdp'])
    print("Coefficient of SDP squared: ", model.params['sdp_sq'])
    print("t-value of SDP: ", model.tvalues['sdp'])
    print("t-value of SDP squared: ", model.tvalues['sdp_sq'])
    print("R-squared: ", model.rsquared)
    sdp.append(model.params['sdp'])
    sdp_sq.append(model.params['sdp_sq'])
    r_sq.append(model.rsquared)


    print("\n")

years = [int(year) for year in years]

# Plot R-squared over the years
plt.figure(figsize=(8, 6))
plt.bar(years, r_sq, color='skyblue', edgecolor='black')
plt.xlabel('Year', fontsize=14)
plt.ylabel('R-squared', fontsize=14)
plt.title('R-squared over the years', fontsize=16)
plt.grid(True)
plt.show()

# Plot Coefficient of SDP over the years
plt.figure(figsize=(8, 6))
plt.bar(years, sdp, color='lightgreen', edgecolor='black')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Coefficient of SDP', fontsize=14)
plt.title('Coefficient of SDP over the years', fontsize=16)
plt.grid(True)
plt.show()







    

