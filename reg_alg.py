from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# Linear regression:
def linearRegression(X_train, X_dev, y_train, y_dev):
    from sklearn import linear_model
    print("Begin linear regression training...")
    model_LinearRegression = linear_model.LinearRegression()
    model_LinearRegression.fit(X_train, y_train)
    y_pred = model_LinearRegression.predict(X_dev)
    RMSE = np.sqrt(mean_squared_error(y_pred, y_dev))

    print ('LogicalClassifier RMSE:', RMSE)

    return model_LinearRegression

# Ridge regression:
def RidgeRegression():
    print("Begin Ridge regression training...")
    return 0

# Lasso regression:
def LassoRegression():
    print("Begin Lasso regression training...")
    return 0

# ElasticNet regression:
def ElasticNetRegression():
    print("Begin ElasticNet regression training...")
    return 0

# ExtraTree regression:
def ExtraTreeRegression():
    print("Begin ExtraTree regression training...")
    return 0

# Decision Tree regression:
def DecisionTreeregression():
    print("Begin Decision Tree regression training...")
    return 0

# Random Forest regression:
def RandomForestregression():
    print("Begin Random Forest regression training...")
    return 0

# Gradient Boosting Random Forest Regression
def GradientBoostingRandomForestRegression():
    print("Begin Gradient Boosting Random Forest regression training...")
    return 0
