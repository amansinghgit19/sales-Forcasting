

Sales Forecasting Project
Overview
This project focuses on forecasting sales using data from an e-commerce platform. The goal is to predict future sales based on product pricing, rating counts, and other features such as weekends and price differences. We employ Linear Regression for forecasting and use various feature engineering techniques, including log transformations and weekend indicators, to improve the model's accuracy.

Features and Insights
Data Cleaning: The dataset is cleaned by removing non-numeric characters (e.g., currency symbols and commas) from columns like discounted_price and rating_count.

Feature Engineering:

Price Difference: Calculated as the difference between the actual_price and discounted_price.
Log Transformation of Rating Count: Used to handle outliers in the rating_count data.
Weekend Indicator: A binary feature indicating whether a date falls on a weekend (Saturday or Sunday).
Model: Linear Regression is employed to predict the total sales revenue (Revenue) based on time-series features like the day of the week and month.

Evaluation: Model performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

Data Visualizations: Key visualizations include sales revenue over time and actual vs predicted sales for better understanding of the model's performance.

Dataset
The dataset is assumed to contain the following columns (adjust according to your actual dataset structure):

product_id: Unique identifier for each product.
product_name: Name of the product.
category: Product category.
discounted_price: Price of the product after discount.
actual_price: Original price of the product before discount.
rating_count: The number of ratings for the product.
Date: Simulated date for each transaction.
Revenue: Calculated as discounted_price * rating_count.
Installation
Clone the repository or download the files to your local machine.

Install the required libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Dataset:

Place the amazon.csv file in the same directory as the script, or adjust the file path accordingly in the script.
Usage
Run the Script:

After installing the necessary dependencies, you can run the script:
bash
Copy code
python sales_forecasting.py
Expected Output:

The script will output the Mean Squared Error (MSE) and Mean Absolute Error (MAE) of the model, showing how well the sales are being predicted.
The script will also display visualizations of the Sales/Revenue over Time and a comparison of Actual vs Predicted Sales.
