import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('merged_data_with_sdp_and_gini.csv')

# Display descriptive statistics for nitrate, sdp, and gini index
print("Descriptive Statistics for Nitrate:")
print(data['nitrate'].describe())
print("Median:", data['nitrate'].median())

print("\nDescriptive Statistics for SDP:")
print(data['sdp'].describe())
print("Median:", data['sdp'].median())

print("\nDescriptive Statistics for Gini Index:")
print(data['gini index'].describe())
print("Median:", data['gini index'].median())

# Plot histograms for nitrate, sdp, and gini index
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(data['nitrate'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Nitrate')
plt.xlabel('Nitrate')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(data['sdp'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Histogram of SDP')
plt.xlabel('SDP')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(data['gini index'], bins=20, color='salmon', edgecolor='black')
plt.title('Histogram of Gini Index')
plt.xlabel('Gini Index')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
