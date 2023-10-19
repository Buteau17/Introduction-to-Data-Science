import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############ Reading Csv file #################

# Reading the Csv file using the Pandas library 
data = pd.read_csv("iris.csv")
print(data.head(10))

######scatter plot -Scatter plots are particularly useful for visualizing the relationship or correlation between two continuous variables

# Extract Sepal.Length and Sepal.Width columns from the iris dataset
sepal_length = data["SepalLength"]
sepal_width = data["SepalWidth"]

# Create the scatter plot
plt.scatter(sepal_length, sepal_width, marker="o", color="blue")

# Set plot title and axis labels
plt.title("Scatterplot of sepal length and speal width")
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

# Display the plot
plt.show()


############Label Data Points in the Scatter plot##########
species = data["Species"]

#print(species)
plt.scatter(sepal_length, sepal_width, marker="o", color="blue")

# Set plot title and axis labels
plt.title("Scatterplot of sepal length and speal width")
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

# Add labels for the first 100 data points
for i in range(100):
    plt.text(sepal_length[i], sepal_width[i], species[i], fontsize=9, color="black", ha='center', va='center')

# Display the plot
plt.show()

###########color all the diffrent species ######
# Create a list of unique species for the legend
unique_species = species.unique()
for sp in unique_species:
    subset = data[data["Species"] == sp]
    #print(subset )
    plt.scatter(subset["SepalLength"], subset["SepalWidth"], marker="o", label=sp)

#Set plot title and axis labels
plt.title("Scatter plot")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# Add legend
plt.legend(loc="upper right")

# Display the plot
plt.show()


###################  Histogram  ##############
petal_length = data["PetalLength"]
# Create a histogram
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.hist(petal_length, bins=5, color='blue', edgecolor='black')

# Set plot title and axis labels
plt.title("Histogram of Petal.Length")
plt.xlabel("Petal.Length")
plt.ylabel("Frequency")

# Display the plot
plt.show()
###########  Bar plot ################

#Calculate the counts of each unique species
species_counts = data["Species"].value_counts()
print(species_counts)
print(species_counts.index)
# Define the colors for each species
colors = {"Setosa": "red", "Virginica": "blue", "Versicolor": "green"}

# Create a bar plot
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.bar(species_counts.index, species_counts, color=[colors[sp] for sp in species_counts.index])

# Set plot title and axis labels
plt.title("Species")
plt.xlabel("Species")
plt.ylabel("Count")

# Add legend
legend_labels = species_counts.index
legend_colors = [colors[sp] for sp in species_counts.index]
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in legend_colors]
plt.legend(legend_handles, legend_labels, loc="upper right")

# Display the plot
plt.show()

###################  Box plot ######################
# Extract the numeric columns (excluding the "species" column)
numeric_columns = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
# Create a box plot
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.boxplot(numeric_columns.values, labels=numeric_columns.columns, patch_artist=True, boxprops=dict(facecolor='red'))

# Set the main title
plt.title("Boxplot")

# Display the plot
plt.show()

############### Density plot ###############

# Create a density plot

# Create a density plot with normal distributions
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.histplot(data=data, x="SepalLength", hue="Species", element="step", common_norm=False, palette="Set1", alpha=0.5, kde=True)

# Set plot title and axis labels
plt.title("Density Plot of Sepal.Length by Species")
plt.xlabel("Sepal Length")
plt.ylabel("Density")

# Display the plot
plt.show()

############# pie chart ##################
# Calculate the counts of each unique species
species_counts = data["Species"].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', colors=['red', 'blue', 'green'])

# Set plot title
plt.title("Pie Chart of Species Distribution")

# Display the plot
plt.show()

################ QQ-Plot¶ ###################
import scipy.stats as stats
# Generate example data (normally distributed)
data = np.random.normal(size=100)

# Create a QQ plot against the normal distribution
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
stats.probplot(data, dist="norm", plot=plt)

# Set plot title and labels
plt.title("QQ Plot Against Normal Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# Display the plot
plt.show()
