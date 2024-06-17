import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# Load the data
data = pd.read_csv('data.csv')

# Analyze missing values
print(data['nitrate'].isnull().sum(), "missing values out of", len(data))


# Plot histogram of nitrate values
plt.hist(data['nitrate'].dropna(), bins=30, alpha=0.7)
plt.title('Distribution of Nitrate Levels')
plt.xlabel('Nitrate')
plt.ylabel('Frequency')
plt.show()


filtered_data = data[['state', 'district', 'year', 'nitrate']]

# Check for missing values in 'nitrate' and fill them using median imputation
if filtered_data['nitrate'].isnull().any():
    imputer = SimpleImputer(strategy='median')
    filtered_data['nitrate'] = imputer.fit_transform(filtered_data['nitrate'].values.reshape(-1, 1))

# filtered_data.to_csv('filtered_data.csv', index=False)

sdp_data = pd.read_csv('sdp_data.csv')

#fill the missing values for Telangana state
imputer = SimpleImputer(strategy='mean')
sdp_data['TELANGANA'] = imputer.fit_transform(sdp_data['TELANGANA'].values.reshape(-1, 1))


# Transpose the DataFrame
transposed_data = sdp_data.set_index('year').T

# Save the transposed DataFrame to a new CSV file
transposed_data.to_csv('transposed_data.csv')
print(transposed_data)
# filtered_data = pd.read_csv('filtered_data.csv')

# Load the SDP data from the new CSV file
# sdp_data_t = pd.read_csv('transposed_data.csv')

