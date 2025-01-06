import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the Amazon dataset
df = pd.read_csv("C:/Users/Gaurav Singh/Desktop/aman project/amazon.csv")  # Update to your local file path

# Clean the `discounted_price` and `rating_count` columns by converting to numeric
df['discounted_price'] = pd.to_numeric(df['discounted_price'].replace({'₹': '', ',': ''}, regex=True), errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'].replace({',': ''}, regex=True), errors='coerce')

# Fill NaN values with 0 for any invalid conversions
df['discounted_price'].fillna(0, inplace=True)
df['rating_count'].fillna(0, inplace=True)

# Simulate a date column assuming a dataset collected in January 2024
df['Date'] = pd.to_datetime(np.random.choice(pd.date_range(start="2024-01-01", end="2024-01-31", freq="D"), len(df)))

# Create a 'Revenue' column based on discounted price and rating count
df['Revenue'] = df['discounted_price'] * df['rating_count']

# Aggregate by date (sum the revenue for each date)
df_grouped = df.groupby('Date').agg({'Revenue': 'sum'}).reset_index()

# Data Visualization: Sales/Revenue over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_grouped, x='Date', y='Revenue')
plt.title('Sales and Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()

# Prepare features and target variable
X = df_grouped[['Date']]  # Date will be the feature
X['day_of_week'] = X['Date'].dt.dayofweek  # Feature engineering: day of the week
X['month'] = X['Date'].dt.month  # Feature engineering: month
y = df_grouped['Revenue']

# Train-test split (80-20), ensure not to shuffle so the time series is preserved
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train[['day_of_week', 'month']], y_train)

# Predict the sales revenue
y_pred = model.predict(X_test[['day_of_week', 'month']])

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot the forecast vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Sales')
plt.plot(y_test.index, y_pred, label='Predicted Sales', linestyle='--')
plt.legend()
plt.title('Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()
