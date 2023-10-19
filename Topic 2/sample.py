import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


file_path = '2023 June Unemployment Rate by County (Percent).csv'
dataset = pd.read_csv(file_path)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

print("First few rows of the DataFrame:")
print(dataset.head(10))
## View the last few rows of the data frame
print(dataset.tail(5))
# Use the type() function to determine the data type of each object
print("Type of the data:", type(dataset))

#Getting the dimension of the data
print(dataset.shape)
#Number of Rows and Columns:
num_rows, num_columns = dataset.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
#Column Names:
column_names = dataset.columns
print("Column names:", column_names)

print("\nSummary statistics:")
print(dataset.describe())

#Check the structure of the data frame
print(dataset.info())
# Use the For loop to get unique value in all the columns
for col_name in dataset.columns:
    #print(data[col_name].unique())
    print("Unique values in column", col_name, ":",dataset[col_name].unique() , "\n")
# Count rows with complete cases (no missing values)
complete_cases_count = dataset.dropna().shape[0]
# Count rows with missing values
missing_cases_count = dataset.shape[0] - complete_cases_count
# Print the counts
print("Rows with complete cases:", complete_cases_count)
print("Rows with missing values:", missing_cases_count)
# Check for missing values and print the count of missing values
missing_values = dataset.isnull().sum()
print("\nMissing values:")
print(missing_values)
missing_value = ["N/a", "na", np.nan]
dataset = pd.read_csv("2023 June Unemployment Rate by County (Percent).csv", na_values = missing_value)
dataset.isnull().sum()
dataset.isnull().any()

# Check for missing values and get the positions
missing_positions = missing_positions = np.where(pd.isna(dataset["Unnamed: 3"]))
print("Positions of missing values in variable  Unnamed: 3 :",missing_positions)

# Use the For loop to get missing values and get the positions in all the columns
for col_name in dataset.columns:
    print("position of the NaN values in all the columns", col_name, ":",np.where(pd.isna(dataset[col_name])) , "\n")



print("\nMissing values:")
print(dataset.isnull().sum())
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

#Get the percentage of missing values in each column
missing_pct = round(dataset.isnull().sum()/len(dataset) * 100, 1)
print(missing_pct)

# Location based replacement
dataset.loc[92, 'Unnamed: 3'] =7.51
dataset.loc[94, 'Unnamed: 3'] = 7.62
dataset.loc[1038, 'Unnamed: 3'] = 7.83
dataset.loc[2419, 'Unnamed: 3'] = 7.94
dataset.loc[2918, 'Unnamed: 3'] = 8.15

print(dataset)
print(dataset['Unnamed: 3'].unique())

# Replace missing values with a number
file_path = '2023 June Unemployment Rate by County (Percent).csv'
dataset = pd.read_csv(file_path)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset['Unnamed: 3'].fillna(8.643, inplace=True)
print(dataset)
print(dataset['Unnamed: 3'].unique())

# Write the DataFrame to a CSV file
dataset.to_csv("new_dataset.csv", index=False)  # Set index=False to omit writing row numbers)