import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Polynomial Regression Salary Predictor")

# Load dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Slider for user input
level = st.slider("Select Position Level", 1.0, 10.0, 6.5)

# Prediction
prediction = lin_reg_2.predict(poly_reg.transform([[level]]))

# Display prediction
st.success(f"Predicted Salary: ₹{int(prediction[0]):,}")

# Create smooth curve
X_grid = np.arange(np.min(X), np.max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# Plot graph
fig, ax = plt.subplots()

ax.scatter(X, y, color="red", label="Actual Salary")

ax.plot(
    X_grid,
    lin_reg_2.predict(poly_reg.transform(X_grid)),
    color="blue",
    label="Polynomial Regression"
)

ax.set_title("Polynomial Regression Fit")
ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.legend()

# Show graph in Streamlit
st.pyplot(fig)
