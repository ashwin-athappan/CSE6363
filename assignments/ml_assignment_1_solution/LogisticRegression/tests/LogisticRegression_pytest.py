import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression.LogisticRegression import LogisticRegression as lgr

def test_lgr():
    iris_data = load_iris()

    petal_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    petal_df['target'] = iris_data.target
    X_lr_pl_pw_sl = petal_df[['sepal length (cm)', 'sepal width (cm)']].values
    y_lr_pl_pw_sl = petal_df['target'].values

    X_lr_pl_pw_sl_train, X_lr_pl_pw_sl_test, y_lr_pl_pw_sl_train, y_lr_pl_pw_sl_test = train_test_split(X_lr_pl_pw_sl, y_lr_pl_pw_sl, test_size=0.1)


    lr_pl_pw_sl = lgr()
    lr_pl_pw_sl.fit(X_lr_pl_pw_sl_train, y_lr_pl_pw_sl_train)

test_lgr()