import utils
import pandas as pd
from sklearn import linear_model, preprocessing


def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

train = pd.read_csv('train.csv')
clean_data(train)

target = train["Survived"].values
feature_names = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = train[feature_names].values

classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)
print classifier_.score(features,target)

poly = preprocessing.PolynomialFeatures(degree = 2)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)
print classifier_.score(poly_features,target)