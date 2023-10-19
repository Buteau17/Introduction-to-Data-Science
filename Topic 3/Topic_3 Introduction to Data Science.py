import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_profiling 
import seaborn as sns
import os

##################  setup the  working directory #################
'''
working_directory = "/Users/kirankumarisheshma/Desktop/Introduction To data science/Topic-3"
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
'''

############ Reading Csv file #################

# Reading the Csv file using the Pandas library 
data = pd.read_csv("survey.csv")
print(data.head(10))

## View the last few rows of the data frame
print(data.tail(5))

# Use the type() function to determine the data type of each object
print("Type of the data:", type(data))

#Getting the dimension of the data
print(data.shape)

#Number of Rows and Columns:
num_rows, num_columns = data.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#Column Names:
column_names = data.columns
print("Column names:", column_names)

#Summary Statistics ((for numeric columns)):
summary_statistics = data.describe()
print("Summary statistics:", summary_statistics)

#Check the structure of the data frame
print(data.info())

#Extract the first 15 rows using iloc
print(data.iloc[0:7])

############## Find Unique Values #######################

# Use pandas unique() to find unique elements in variable "sex"
unique_elements = data['sex'].unique()
print("Unique values in variables sex:", unique_elements)

# Unique elements in variable "height"
unique_elements = data['height'].unique()
print("Unique values in variables height:", unique_elements)


# Use the For loop to get unique value in all the columns
for col_name in data.columns: 
    #print(data[col_name].unique())
    print("Unique values in column", col_name, ":",data[col_name].unique() , "\n")

########################## Find Missing Values Or NAs #############################

# Count rows with complete cases (no missing values)
complete_cases_count = data.dropna().shape[0]           #data.dropna() to create a new DataFrame with rows that have no missing values. .shape[0] to count the number of rows in this DataFrame, representing the rows with complete cases.
# Count rows with missing values
missing_cases_count = data.shape[0] - complete_cases_count
# Print the counts
print("Rows with complete cases:", complete_cases_count)
print("Rows with missing values:", missing_cases_count)

   
# count of missing values for each row
missing_rows_count = data.isna().sum(axis=0)
# Print the count of missing values for each row
print("Count of missing values in each row:","\n", missing_rows_count, )


# Check for missing values and get the positions
missing_positions = missing_positions = np.where(pd.isna(data["sex"]))
print("Positions of missing values in variable sex:",missing_positions)


# Use the For loop to get missing values and get the positions in all the columns
for col_name in data.columns: 
    print("position of the NaN values in all the columns", col_name, ":",np.where(pd.isna(data[col_name])) , "\n")



############## Cleaning the dataset #################
   
data['sex'] = data['sex'].replace('F', 'Female')           # replace 'F" to 'Female' 
data['sex'] = data['sex'].replace('M', 'Male')           #'M' to 'Male' in the 'sex' variable
# Print the updated DataFrame
print(data)
print(data['sex'].unique())


################## Replace continuous data with mean in height variable #########################

# Calculate the mean of female heights (ignoring missing values)
female_height_mean = data[data['sex'] == 'Female']['height'].mean()
print("Mean of Female Heights:", female_height_mean)

# Calculate the mean of male heights (ignoring missing values)
male_height_mean = data[data['sex'] == 'Male']['height'].mean()
print("Mean of Female Heights:", male_height_mean)


# Replace missing values (NaN) in the 'height' column
data.loc[(data['sex'] == 'Female') & (data['height'].isna()), 'height'] = female_height_mean
data.loc[(data['sex'] == 'Male') & (data['height'].isna()), 'height'] = male_height_mean
print(np.where(pd.isna(data["height"])))
print(data['height'])


################# Replace categorical data with bootstrap in Python #################

def bootstrap(series):
    # Filter the series to keep only complete cases
    complete_series = series.dropna()
    
    # Calculate the probability distribution based on frequencies
    tb = pd.Series(complete_series).value_counts(normalize=True)
    
    # Generate a sample based on the probability distribution
    smpl = np.random.choice(tb.index, size=len(series) - len(complete_series), p=tb.values)
    
    # Replace missing values in the original series with the sample
    series[series.isna()] = np.random.choice(smpl, size=len(series) - len(complete_series))
    
    return series
data['handedness'] = bootstrap(data['handedness'] )
data['sex'] = bootstrap(data['sex'])
data['smoke'] = bootstrap(data['smoke'])
#print(data)

# Use the For loop to get missing values and get the positions in all the columns
for col_name in data.columns: 
    print("position of the NaN values in all the columns", col_name, ":",np.where(pd.isna(data[col_name])) , "\n")


# Write the DataFrame to a CSV file
data.to_csv("new_survey.csv", index=False)  # Set index=False to omit writing row numbers




