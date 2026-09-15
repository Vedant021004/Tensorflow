import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([
    [1000],
    [1500],
    [2000],
    [2500]
])

y = np.array([
    200000,
    300000,
    400000,
    500000
])

# Create model
model = LinearRegression()

# Train
model.fit(X, y)

# Predict
prediction = model.predict([[3000]])

print("Predicted price:", prediction[0])

# Learned parameters
print("Weight:", model.coef_[0])
print("Bias:", model.intercept_)