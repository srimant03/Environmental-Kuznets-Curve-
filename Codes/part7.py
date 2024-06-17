import pandas as pd

regions = {
    "Northern Region": ["CHANDIGARH", "DELHI", "HARYANA", "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "PUNJAB", "RAJASTHAN"],
    "North-Eastern Region": ["ARUNACHAL PRADESH", "ASSAM", "MANIPUR", "MEGHALAYA", "NAGALAND", "TRIPURA"],
    "Eastern Region": ["ANDAMAN AND NICOBAR ISLANDS", "BIHAR", "JHARKHAND", "ODISHA", "WEST BENGAL"],
    "Central Region": ["CHHATTISGARH", "MADHYA PRADESH", "UTTAR PRADESH", "UTTARAKHAND"],
    "Western Region": ["GOA", "GUJARAT", "MAHARASHTRA"],
    "Southern Region": ["ANDHRA PRADESH", "KARNATAKA", "KERALA", "pondicherry", "TAMIL NADU"]
}

northern_region = regions["Northern Region"]
northeastern_region = regions["North-Eastern Region"]
eastern_region = regions["Eastern Region"]
central_region = regions["Central Region"]
western_region = regions["Western Region"]
southern_region = regions["Southern Region"]

# Read the data
data = pd.read_csv('merged_data_with_sdp_and_gini_updated.csv')

# Check for null values in gini index
print(data['gini index'].isnull().sum())

# Drop null values of gini index
data = data.dropna(subset=['gini index'])

# Do a region wise regression analysis
import statsmodels.api as sm

for region, states in regions.items():
    print("Region: ", region)
    data_region = data[data['state'].str.upper().isin(states)]
    X = data_region[['sdp', 'gini index']]
    X['sdp_sq'] = X['sdp']**2
    #X['sdp_gini_interaction'] = X['sdp'] * X['gini index']
    X = sm.add_constant(X)
    y = data_region['nitrate']

    model = sm.OLS(y, X).fit()
    predicted_values = model.predict(X)

    print("Coefficient of SDP: ", model.params['sdp'])
    print("Coefficient of SDP squared: ", model.params['sdp_sq'])
    print("Coefficient of Gini Index: ", model.params['gini index'])
    print("t-value of SDP: ", model.tvalues['sdp'])
    print("t-value of SDP squared: ", model.tvalues['sdp_sq'])
    print("t-value of Gini Index: ", model.tvalues['gini index'])
    print("R-squared: ", model.rsquared)

    print("\n")

