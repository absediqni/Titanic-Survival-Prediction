#Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

#load dataset
df = pd.read_csv("train.csv")
df.head()

# EDA
#Check missing values
df.isnull().sum()
# Visualise survival rate 
sns.countplot(x="Survived", data=df)
sns.countplot(x="Sex", hue="Survived", data=df)
sns.countplot(x="Pclass", hue="Survived", data=df)

# Data processing
# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Encode Categorical variables
data = pd.get_dummies(df, columns=["Sex", "Embarked"])

#Feature Engineering
#Create family size
data["FamilySize"] = df["SibSp"] + df["Parch"] + 1

#Model Training
features = ["Pclass", "Age", "Fare", "FamilySize", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]
X = data[features]
y = data["Survived"]
#Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Train a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#Predict
y_pred = model.predict(X_test)

#Model Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Feature Importance
importances = model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()


#INTERMEDIATE

#Heatmap of survival by Age and class
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", palette="Set2")
plt.title("Survival Distribution by Age")

#Survival by class and sex
sns.catplot(x="Pclass", hue="Survived", col="Sex", kind="count", data=df)

#MORE FEATURE ENGINEERING
#Extract dect form cabin
df["Deck"] = df["Cabin"].str[0]
df["Deck"].fillna("Unknown")

# Extract titles from name
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\\.', expand=False)

#Group rare titles
rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Rev", "Sir", "Jonkheer"]
df["Title"] = df["Title"].replace(rare_titles, "Rare")

#MODEL COMPRESSION

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, clf.predict(X_test))

pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])

#Hyper Parameter tunning
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

#Model Explainer
explainer = shap.TreeExplainer(grid.best_estimator_)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=features)

#Interactivity Widgets

import ipywidgets as widgets
from ipywidgets import interact

def survival_by_age(age_cutoff=18):
    subset = df[df["Age"] <= age_cutoff]
    sns.countplot(x="Sex", hue="Survived", data=subset)
    plt.title(f"Survival for Age <= {age_cutoff}")
    plt.show()

widgets.interact(survival_by_age, age_cutoff=(0,80))
