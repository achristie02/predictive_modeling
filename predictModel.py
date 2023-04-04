import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from scipy import stats

# Read in the dataset
data = pd.read_csv('PredictiveModelingAssessmentData.csv')

# Split the dataset into x and y variables
x = data[['x1', 'x2']]
y = data['y']

# Calculate Z-scores for y
y_z_scores = np.abs(stats.zscore(y))

# Find outliers in y
y_outliers = np.where(y_z_scores > 2.5)

# Remove rows with y outliers
x_clean = np.delete(x.values, y_outliers, axis=0)
y_clean = np.delete(y.values, y_outliers, axis=0)

models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(max_iter=1000),
          SVR()]

# Define the RMSE scorer for cross-validation
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Fit and test each model using 5-fold cross-validation
rmse_scores = []
for model in models:
    # Calculate the cross-validated RMSE for the model
    cv_scores = cross_val_score(model, x_clean, y_clean, cv=5, scoring=rmse_scorer)
    
    # Print the average RMSE across all folds
    print(type(model).__name__, "RMSE:", np.mean(cv_scores))
    
    # Add the average RMSE to the list of RMSE scores
    rmse_scores.append(np.mean(cv_scores))

# Get the index of the model with the lowest RMSE
best_model_idx = np.argmin(rmse_scores)

# Use the best model to make predictions on the test dataset
best_model = models[best_model_idx]
best_model.fit(x_clean, y_clean)  # Fit the model on the entire cleaned training dataset
test_data = pd.read_csv('TestData.csv')
x_test = test_data[['x1', 'x2']]
predictions = best_model.predict(x_test)
print(predictions[0])
# Save the predictions to a csv file or print them out
pd.DataFrame({'y': predictions}).to_csv('TestDataPredictions.csv', index=False)

print(predictions)
