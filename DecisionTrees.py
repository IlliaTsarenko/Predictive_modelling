from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# Prepare the data
X = investment_data.drop('target')
y = investment_data['target']
# Create a decision tree regressor
model = DecisionTreeRegressor()

# Define the hyperparameters to tune
param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the model performance
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)

print("Best Model:", best_model)
print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)