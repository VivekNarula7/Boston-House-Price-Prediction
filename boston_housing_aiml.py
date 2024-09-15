# Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# We load are dataset from csv file into a pandas dataframe
df = pd.read_csv('/home/vivek/Downloads/boston-housing/boston_housing_dataset/train.csv')
print(df)

## Checks for the presence of null values in the dataset
any_nulls = df.isnull().values.any()
print(any_nulls)  # Returns True if there are any null values, False otherwise

# We drop the 'ID' attribute as this is redundant and does not contribute to the predcition of our target variable
df.drop('ID', axis=1, inplace=True)
print(df)

# We store the target variable into a separate variable and try to run our predciction operations on this
X = df.drop('medv', axis=1)
y = df['medv'] # the target variable

print(X,y)

# Splitting the dataset into training and testing set using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# We train our Linear Regression model using sklearn.linear_model.LinearRegression() 
model = LinearRegression()
model.fit(X_train, y_train) # fitting the model using the fit method

# Evaluating the model on the test set
test_score = model.score(X_test, y_test) # The score function built into the LinearRegression module computes the R2 score
print(f"Test R² Score: {test_score:.4f}") 

y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score (Test): {r2:.4f}")
