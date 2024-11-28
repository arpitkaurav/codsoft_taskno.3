import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('advertising.csv')
df.fillna(method='ffill', inplace=True)
df = pd.get_dummies(df, drop_first=True)

X, y = df.drop('Sales', axis=1), df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae, mse, r2 = mean_absolute_error(y_test, predictions), mean_squared_error(y_test, predictions), r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", color="blue")
plt.plot(predictions, label="Predicted", color="red", linestyle="--")
plt.legend()
plt.show()
