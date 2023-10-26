# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi():
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    train_data = pd.read_csv("data/titanic/train.csv")
    test_data = pd.read_csv("data/titanic/test.csv")

    y = train_data["Sex"].map({"male": 0, "female": 1})

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = train_data[features]
    # data['Embarked'] = data['Embarked'].fillna('S')

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # data['Age'] = imp.fit_transform(data[['Age']])

    model = DecisionTreeClassifier()
    model.fit(X, y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
