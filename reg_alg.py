from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_model(model, parameter, X, y):
    grid_model = GridSearchCV(estimator=model,param_grid=parameter,scoring="r2",cv=5)
    grid_model.fit(X,y)
    return grid_model.best_score_, grid_model.best_estimator_

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
def RidgeRegression(X, y):
    from sklearn import linear_model
    print("Begin Ridge regression training...")
    alpha = np.logspace(-6,-1,10)
    parameter = dict(alpha=alpha)
    model = linear_model.Ridge()
    return train_model(model,parameter,X,y)

# Lasso regression:
def LassoRegression():
    print("Begin Lasso regression training...")
    return 0

# ElasticNet regression:
def ElasticNetRegression():
    print("Begin ElasticNet regression training...")
    return 0

# ExtraTree regression:
def ExtraTreeRegression(X, y):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import ExtraTreeRegressor
    print("Begin ExtraTree regression training...")
    n_estimators = np.array([i for i in range(5,15)])
    parameter = dict(n_estimators=n_estimators)
    extra_tree = ExtraTreeRegressor(random_state=0)
    model = BaggingRegressor(extra_tree, random_state=0)

    return train_model(model,parameter,X,y)

# Decision Tree regression:
def DecisionTreeregression(X, y):
    from sklearn import tree
    print("Begin Decision Tree regression training...")
    max_depth = np.logspace(0,1,10)
    parameter = dict(max_depth=max_depth)
    model = tree.DecisionTreeRegressor()

    return train_model(model,parameter,X,y)

# Random Forest regression:
def RandomForestregression():
    print("Begin Random Forest regression training...")
    return 0

# Gradient Boosting Random Forest Regression
def GradientBoostingRandomForestRegression():
    print("Begin Gradient Boosting Random Forest regression training...")
    return 0
