'''
OUESTION

Using the pandas function read_csv(), read the given ‘iris’ data set and perform the
following:
a. Shape of the data set.
b. First 5 and last five rows of data set (head and tail).
c. Size of data set.
d. No. of samples available for each variety.
e. Description of the data set( use describe)
Using the iris data set perform seaborn - pairplot() and displot() functions'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
iris=pd.read_csv("iris.csv")
print("Shape of the Data set :",iris.shape)
print("First five rows")
print(iris.head())
print("***************")
print("Last five rows")
print(iris.tail())
print("Size of the Data Set :",iris.size)
print("Number of samples available for each Variety")
print(iris["variety"].value_counts())
print("Description of the data set")
print(iris.describe())
sns.pairplot(iris,hue="variety", kind="scatter",diag_kind="hist")
plt.style.use("dark_background")
sns.displot(iris.sepal_length,bins=10, color="g")
plt.title("Distribution of Sepal Length", fontsize=10, color="white")
plt.show()
