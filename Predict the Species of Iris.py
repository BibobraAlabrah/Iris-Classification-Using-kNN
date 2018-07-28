
# coding: utf-8

# # Prediction of Iris Species Using kNN 
# Bibobra M. Alabrah

# INTRODUCTION
# 
# The main goal of this project is to use machine learning to predict the species of the iris flowers given a set of features.
# The data description is presented in the data profile section. This is s three-class classification problem because the flowers have three species namely: setosa, versicolour, and virginica.

# In[1]:


# IMPORT DEPENDENCIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[2]:


# LOAD THE DATASET

iris_data = load_iris()


# In[3]:


# MEET THE DATASET

print("Keys of iris_data: \n{}".format(iris_data.keys()))


# In[4]:


# DESCR is the dataset description: Let us take a look at it.

print(iris_data['DESCR'][:987])


# In[5]:


# Just to reaffirm the data distribution or shape

print("Shape of Data: {}".format(iris_data['data'].shape))


# The dataset contains 150 flowers and 4 features or variables.

# In[6]:


# VIEW THE FIRST 5 ROWS OF THE DATASET
print("The First 5 Rows of the Dataset: \n{}".format(iris_data['data'][:5]))


# From this data, we can see that all of the first five flowers have a petal width of 0.2 cm
# and that the first flower has the longest sepal, at 5.1 cm.

# In[7]:


# view the target
print("The Target: \n{}".format(iris_data['target']))


# The meanings of the numbers are given by the iris['target_names'] array:
# 0 means setosa, 1 means versicolour, and 2 means virginica.

# In[8]:


# Let us split the data into the train and test set to build the model and also test its effectiveness
from sklearn.model_selection import train_test_split 


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(iris_data['data'], iris_data['target'], random_state = 0)


# X_train contains 75% of the rows of the dataset,
# and X_test contains the remaining 25%

# In[10]:


# Let us view the dimensions of the new datasets
print("X_train Shape: {}".format(X_train.shape))
print("Y_train Shape: {}".format(Y_train.shape))
print("X_test Shape: {}".format(X_test.shape))
print("Y_test Shape: {}".format(Y_test.shape))


# In[11]:


# Let us use some visualization to understand our data. First convert the numpy arrays to a dataframe using pandas
iris_df = pd.DataFrame(X_train, columns = iris_data.feature_names)


# In[12]:


iris_df.head()


# In[13]:


# create a scatter matrix from the dataframe, color by Y_train
plot_1 = pd.plotting.scatter_matrix(iris_df, c = Y_train, figsize = (15, 15), marker = 'o', 
                                    hist_kwds = {'bins': 20}, s=60, alpha=0.8)


# Plot_1. Pair plot of the Iris dataset, colored by class label

# From the plots, we can see that the three classes seem to be relatively well separated
# using the sepal and petal measurements. This means that a machine learning model
# will likely be able to learn to separate them.

# # Building the model

# In[14]:


# Build the model using k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier


# In[15]:


# Instantiate the class into an object
knn = KNeighborsClassifier(n_neighbors = 3)


# To build the model on the training set, we call the fit method of the knn object,
# which takes as arguments the NumPy array X_train containing the training data and
# the NumPy array y_train of the corresponding training labels:

# In[16]:


knn.fit(X_train, Y_train)


# # Making Predictions
# 
# Imagine we found an iris in the wild with a sepal length of
# 5 cm, a sepal width of 2.9 cm, a petal length of 1 cm, and a petal width of 0.2 cm.
# What species of iris would this be?

# In[17]:


# Create this new data and store in a numpy array format
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))


# In[18]:


# Predict the species of iris this would be.

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_data['target_names'][prediction]))


# Our model predicts that this new iris belongs to the class 0, meaning its species is
# setosa. But how do we know whether we can trust our model? We don’t know the correct
# species of this sample, which is the whole point of building the model!

# # Model Evaluation

# This is where the test set that we created earlier comes in. This data was not used to
# build the model, but we do know what the correct species is for each iris in the test
# set.
# Therefore, we can make a prediction for each iris in the test data and compare it
# against its label (the known species). We can measure how well the model works by
# computing the accuracy, which is the fraction of flowers for which the right species
# was predicted

# In[19]:


Y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(Y_pred))


# In[20]:


# We can use the score method of the knn object, which will compute the prediction accuracy for us:
print("Prediction Accuracy: {:.2f}".format(knn.score(X_test, Y_test)))


# For this model, the test set accuracy is about 0.97, which means we made the right
# prediction for 97% of the irises in the test set. Under some mathematical assumptions,
# this means that we can expect our model to be correct 97% of the time for new
# irises.
