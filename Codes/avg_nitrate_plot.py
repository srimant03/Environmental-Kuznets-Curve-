import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
# Assuming your dataset is stored in a CSV file named 'merged_data_with_sdp_and_gini.csv'
data = pd.read_csv('merged_data_with_sdp_and_gini.csv')

# Group the data by year and calculate the average nitrate value for each year
average_nitrate_by_year = data.groupby('year')['nitrate'].mean()

# Plot the average nitrate values by year
plt.figure(figsize=(10, 6))
ax = average_nitrate_by_year.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Nitrate Levels by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Nitrate Levels (mg/L)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate each bar with the value it represents
for i, v in enumerate(average_nitrate_by_year):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

