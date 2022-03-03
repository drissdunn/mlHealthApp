import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the csv file
df = pd.read_csv("pima-data.csv")

a = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(a)

# Select independent and dependent variable
x = df.drop(['diabetes', 'skin'], axis = 1).values
y = df.loc[:, "diabetes"].values

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Instantiate the model
rf = RandomForestClassifier()

# Fit the model
rf.fit(x_train, y_train)

# pickle file of the model
pickle.dump(rf, open("model.pkl", "wb"))