import pandas as pd

'''data_T1 = pd.read_csv('sdp_table1.csv')
data_T2 = pd.read_csv('sdp_table2.csv')

print(data_T1.head())
print(data_T2.head())

#for each column in data_T2 starting from the second column
for column in data_T2.columns[1:]:
    x = data_T2[column][0]
    y = data_T1[column].iloc[-1]
    adj_factor = x/y
    z = data_T1[column]*adj_factor
    data_T1[column] = z

print(data_T2.head())
data = pd.concat([data_T1, data_T2], ignore_index=True)
print(data.head())
print(data.tail())
print(data.shape)
print(data)
#remove 4th row
data = data.drop([4])
print(data)

data_T3 = pd.read_csv('sdp_table3.csv')
print(data_T3.head())

for column in data_T3.columns[1:]:
    x = data_T3[column][0]
    y = data[column].iloc[-1]
    adj_factor = x/y
    z = data[column]*adj_factor
    data[column] = z

print(data_T3.head())
data = pd.concat([data, data_T3], ignore_index=True)
print(data.head())
print(data.tail())
print(data.shape)
print(data)
#remove 12th row
data = data.drop([12])

print(data)

data.to_csv('sdp_data_merged.csv', index=False)'''

#read the merged data
data = pd.read_csv('sdp_data_merged.csv')
print(data.head())

#transpose the data
transposed_data = data.set_index('YEAR').T

#save the transposed data
transposed_data.to_csv('sdp_transposed_data.csv')
