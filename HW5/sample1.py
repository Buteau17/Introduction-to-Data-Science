import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
sns.set(color_codes=True)
file_path = 'cybersecurity_attacks.csv'
dataset = pd.read_csv(file_path)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)
print(dataset.head(10))
# Use the type() function to determine the data type of each object
print("Type of the data:", type(dataset))
## View the last few rows of the data frame
print(dataset.tail(5))

#Getting the dimension of the data
print(dataset.shape)

duplicate_rows_df = dataset[dataset.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)

dataset.count()

dataset = dataset.drop_duplicates()
dataset.head(5)

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
dataset = pd.read_csv('cybersecurity_attacks.csv', na_values = missing_value)
dataset.isnull().sum()
dataset.isnull().any()

# Use the For loop to get missing values and get the positions in all the columns
for col_name in dataset.columns:
    print("position of the NaN values in all the columns", col_name, ":",np.where(pd.isna(dataset[col_name])) , "\n")

print("\nMissing values:")
print(dataset.isnull().sum())
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

missing_pct = round(dataset.isnull().sum()/len(dataset) * 100, 1)
print(missing_pct)

# for col_name in dataset.columns:
data = dataset.dtypes
for i, (key , value) in enumerate (data.items()):
  if value.name == 'int64' or value.name== 'float64':
    plt.figure(i)
    sns.boxplot(x=dataset[key])
    print(key)
    print(value)
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
dataset.shape