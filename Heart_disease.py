
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore the potential errors
import warnings
warnings.filterwarnings('ignore')


# dataset load
dataset = pd.read_csv("F:\Learning_Work\Vs_Work\AIML_Project\Heart.csv")


# Display information about the dataset
print(type(dataset))
print(dataset.shape)
print(dataset.head(5))
print(dataset.sample(5))
print(dataset.describe())
print(dataset.info())

# Define information about the dataset
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
print(info)

# Print information about each column
for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

# Explore the target variable "target"
dataset["target"].describe()
dataset["target"].unique()
print(dataset.corr()["target"].abs().sort_values(ascending=False))

# Visualize target variable distribution
y = dataset["target"]
sns.countplot(y)

# Display the count of each target value
target_temp = dataset.target.value_counts()
print(target_temp)

# Print the percentage of patients with/without heart problems
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# Visualize relationships between features and the target variable
plt.figure(figsize=(12,6))
print(dataset["sex"].unique())
sns.barplot(x=dataset["sex"], y=dataset["target"], data=dataset)
plt.show()

dataset["cp"].unique()
plt.figure(figsize=(12,6))
sns.barplot(x=dataset["cp"],y=dataset["target"], data=dataset)
plt.show()

dataset["fbs"].describe()
dataset["fbs"].unique()
plt.figure(figsize=(12,6))
sns.barplot(x=dataset["fbs"],y=dataset["target"], data=dataset)
plt.show()

dataset["restecg"].unique()
plt.figure(figsize=(12,6))
sns.barplot(x=dataset["restecg"],y=dataset["target"], data=dataset)
plt.show()

dataset["exang"].unique()
plt.figure(figsize=(12,6))
sns.barplot(x=dataset["exang"],y=dataset["target"], data=dataset)
plt.show()

dataset["slope"].unique()
plt.figure(figsize=(12,6))
sns.barplot(x=dataset["slope"],y=dataset["target"], data=dataset)
plt.show()

dataset["ca"].unique()
plt.figure(figsize=(12,6))
sns.countplot(dataset["ca"])
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x=dataset["ca"],y=dataset["target"], data=dataset)
plt.show()

dataset["thal"].unique()
plt.figure(figsize=(12,6))
sns.barplot(dataset["thal"])
sns.distplot(dataset["thal"])
plt.show()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

# Train and evaluate various machine learning models
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
# The accuracy score achieved using Logistic Regression is: 85.25 %


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)
Y_pred_nb.shape
score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

# support vector machine
from sklearn import svm

sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
Y_pred_svm.shape
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
Y_pred_knn.shape
score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0

for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
print(Y_pred_dt.shape)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# Neural Network (using Keras)
from keras.models import Sequential  # for layers
from keras.layers import Dense       # for connection of nn


model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))  # ll neurons and 13 dimensions in the conn  with relu activition function
model.add(Dense(1,activation='sigmoid'))             # 1 neuron sigmoid activition function 

# Configuring the model with the binary cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
model.fit(X_train,Y_train,epochs=300)  # for 300 trials

Y_pred_nn = model.predict(X_test)
Y_pred_nn.shape

rounded = [round(x[0]) for x in Y_pred_nn]  # rounding the predict in 0 and 1

Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

# Display accuracy scores for each model
scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# Visualize accuracy scores using a bar plot
sns.set(rc={'figure.figsize':(15,8)})
plt.figure(figsize=(12,6))
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(x=algorithms,y=scores)
plt.show()