from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./Advertising.csv')

# Features and target variable
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Make sure the same scaler fitting is applied to the test data.
X_test_scaled = scaler.transform(X_test)

# Initialize the neural network regressor with modified parameters
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, learning_rate_init=0.001, random_state=42)
# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the predicted vs actual sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


