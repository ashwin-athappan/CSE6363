import pandas as pd
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression as LR

def test_LR():
    # Given petal length and petal width, predict sepal length.
    petal_iris_data = load_iris()

    petal_df = pd.DataFrame(petal_iris_data.data, columns=petal_iris_data.feature_names)
    petal_X = petal_df['petal length (cm)'].values
    petal_y = petal_df['petal width (cm)'].values

    split_index = int(0.9 * len(petal_X))
    petal_X_train, petal_X_test = petal_X[:split_index], petal_X[split_index:]
    petal_y_train, petal_y_test = petal_y[:split_index], petal_y[split_index:]

    lr_pl_pw = LR()
    lr_pl_pw.fit(petal_X_train, petal_y_train)
    print(petal_X.shape)

test_LR()
