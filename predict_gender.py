import pandas as pd
import clean

# reading and cleaning data
train = pd.read_csv('train.csv')
clean.clean_data(train)
test = pd.read_csv('test.csv')

# first model (all women survive, all men die)
# independent of training set
test["survived_prediction"] = 0
test.loc[test.Sex == "female", "survived_prediction"] = 1

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test["survived_prediction"]})
submission.to_csv("titanic_gender.csv", index = False)