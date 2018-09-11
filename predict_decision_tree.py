import pandas as pd
from sklearn import tree, model_selection
import clean

# reading & cleaning data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
clean.clean_data(train)

# extracting data from training and test sets
y_train_all = train["Survived"]
features = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
X_train_all = train[features]
X_test_all = test[features]

# extracts data for cross-validation
X_train, X_val, y_train, y_val= model_selection.train_test_split(X_train, y_train, random_state=1)

# decision tree model
model = tree.DecisionTreeClassifier(random_state = 1,
                                            max_depth = 7,
                                            min_samples_split=2)
model_sample = tree.DecisionTreeClassifier()

# fits model for sample training
model_sample.fit(X_train, y_train)
# fits model for entire training
model.fit(X_train_all, y_train_all)

# prediction based on test set
prediction_sample = model_sample.predict()
prediction = model.predict(X_test_all)

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("titanic_decisiontree.csv", index = False)
