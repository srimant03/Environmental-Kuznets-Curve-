import pandas as pd
from sklearn.impute import SimpleImputer

transposed_data=pd.read_csv('sdp_transposed_data.csv')

filtered_data=pd.read_csv('filtered_data.csv')

# Melt the SDP data to reshape it
melted_sdp_data = transposed_data.melt(id_vars=['state'], var_name='year', value_name='sdp')
#print number of rows
print(len(melted_sdp_data))

# Convert state names to lowercase for case-insensitive comparison
filtered_data['state'] = filtered_data['state'].str.lower()
melted_sdp_data['state'] = melted_sdp_data['state'].str.lower()

# Convert the 'year' column in melted_sdp_data to integer
melted_sdp_data['year'] = melted_sdp_data['year'].astype(int)


# Merge the melted SDP data into filtered_data based on 'state' and 'year' columns
merged_data_with_sdp = pd.merge(filtered_data, melted_sdp_data, on=['state', 'year'], how='left')
print(len(merged_data_with_sdp))
# Save the merged DataFrame to a new CSV file
# merged_data.to_csv('merged_data_with_sdp.csv', index=False)


# merged_data_with_sdp = pd.read_csv('merged_data_with_sdp.csv')

merged_data_with_sdp.dropna(inplace=True)
print(len(merged_data_with_sdp))
# Load the Gini index data from the new CSV file
gini_data = pd.read_csv('gini_index.csv')

# Convert district names to lowercase for case-insensitive comparison
merged_data_with_sdp['district'] = merged_data_with_sdp['district'].str.lower()
gini_data['district'] = gini_data['district'].str.lower()

# Merge the Gini index data into merged_data_with_sdp based on 'district' column
merged_data_with_sdp = pd.merge(merged_data_with_sdp, gini_data, on='district', how='left')
print(len(merged_data_with_sdp))

# Save the updated DataFrame to a new CSV file
merged_data_with_sdp.to_csv('merged_data_with_sdp_and_gini_updated.csv', index=False)

