# EX-06-Implement-Holt-Winters-method-in-Python.
# AIM:
To implement the Holt-Winters method in Python for time series forecasting and analyze the trend and seasonal patterns in the given dataset.
# ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
# PROGRAM:
```
# ============================================================
# BMW SALES TIME SERIES FORECASTING
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv('/content/BMW sales data (2010-2024).csv')

print("First 5 rows:")
print(data.head())

print("\nDataset columns:")
print(data.columns)


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 4. CREATE YEARLY SALES DATA
# ============================================================

# Sum Sales_Volume for each year
data_yearly = data.groupby('Year')['Sales_Volume'].sum()

# Convert Year into datetime index
data_yearly.index = pd.to_datetime(
    data_yearly.index.astype(str),
    format='%Y'
)

# Set yearly frequency
data_yearly = data_yearly.asfreq('YS')

print("\nYearly BMW Sales:")
print(data_yearly)


# ============================================================
# 5. PLOT ORIGINAL YEARLY DATA
# ============================================================

plt.figure(figsize=(12, 5))

data_yearly.plot()

plt.title('BMW Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Sales Volume')
plt.grid(True)

plt.show()


# ============================================================
# 6. MIN-MAX SCALING
# ============================================================

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_yearly.values.reshape(-1, 1)
    ).flatten(),
    index=data_yearly.index
)

print("\nScaled data:")
print(scaled_data)


# ============================================================
# 7. PLOT SCALED DATA
# ============================================================

plt.figure(figsize=(12, 5))

scaled_data.plot()

plt.title('Scaled BMW Sales')
plt.xlabel('Year')
plt.ylabel('Scaled Sales')
plt.grid(True)

plt.show()


# ============================================================
# 8. SEASONAL DECOMPOSITION
# ============================================================

# BMW data is yearly, so there is no monthly seasonality.
# Period=3 is used only to demonstrate decomposition.

decomposition = seasonal_decompose(
    data_yearly,
    model='additive',
    period=3
)

plt.figure(figsize=(12, 8)) # Moved figsize here
decomposition.plot()

plt.suptitle(
    'BMW Sales Time Series Decomposition',
    fontsize=14
)

plt.show()


# ============================================================
# 9. SHIFT DATA FOR MULTIPLICATIVE MODEL
# ============================================================

# Multiplicative Holt-Winters requires positive values.
# Min-Max scaled values contain 0, so add 1.

scaled_data_positive = scaled_data + 1

print("\nPositive scaled data:")
print(scaled_data_positive)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

# Use 80% for training and 20% for testing

split_point = int(len(scaled_data_positive) * 0.8)

train_data = scaled_data_positive.iloc[:split_point]

test_data = scaled_data_positive.iloc[split_point:]

print("\nTraining data:")
print(train_data)

print("\nTesting data:")
print(test_data)


# ============================================================
# 11. MULTIPLICATIVE HOLT-WINTERS MODEL
# ============================================================

model_mul = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=3
)

fit_mul = model_mul.fit()

print("\nModel fitted successfully.")


# ============================================================
# 12. FORECAST TEST DATA
# ============================================================

test_predictions_mul = fit_mul.forecast(
    steps=len(test_data)
)

print("\nTest Predictions:")
print(test_predictions_mul)


# ============================================================
# 13. VISUAL EVALUATION
# ============================================================

plt.figure(figsize=(12, 6))

ax = train_data.plot(
    label='Train Data'
)

test_predictions_mul.plot(
    ax=ax,
    label='Test Predictions'
)

test_data.plot(
    ax=ax,
    label='Test Data'
)

plt.title('Visual Evaluation - BMW Sales Forecast')
plt.xlabel('Year')
plt.ylabel('Scaled Sales')
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 14. CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_mul
    )
)

mae = mean_absolute_error(
    test_data,
    test_predictions_mul
)

mse = mean_squared_error(
    test_data,
    test_predictions_mul
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)


# ============================================================
# 15. STANDARD DEVIATION AND MEAN
# ============================================================

print("\n==============================")
print("DATA STATISTICS")
print("==============================")

print("Standard Deviation:",
      np.sqrt(scaled_data_positive.var()))

print("Mean:",
      scaled_data_positive.mean())


# ============================================================
# 16. FINAL MODEL USING COMPLETE DATA
# ============================================================

final_model = ExponentialSmoothing(
    scaled_data_positive,
    trend='add',
    seasonal='mul',
    seasonal_periods=3
)

final_fit = final_model.fit()

print("\nFinal model fitted successfully.")


# ============================================================
# 17. FORECAST NEXT 5 YEARS
# ============================================================

forecast_steps = 5

final_predictions = final_fit.forecast(
    steps=forecast_steps
)

print("\n==============================")
print("FUTURE FORECAST")
print("==============================")

print(final_predictions)


# ============================================================
# 18. PLOT FINAL FORECAST
# ============================================================

plt.figure(figsize=(12, 6))

scaled_data_positive.plot(
    label='Historical BMW Sales'
)

final_predictions.plot(
    label='Future Predictions',
    marker='o'
)

plt.title('BMW Sales Prediction - Next 5 Years')
plt.xlabel('Year')
plt.ylabel('Scaled Sales')
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 19. CONVERT FORECAST BACK TO ORIGINAL SALES SCALE
# ============================================================

# Remove the +1 shift
forecast_without_shift = final_predictions - 1

# Convert back from Min-Max scaled values
original_forecast = scaler.inverse_transform(
    forecast_without_shift.values.reshape(-1, 1)
).flatten()

# Create forecast table
forecast_table = pd.DataFrame({
    'Year': final_predictions.index.year,
    'Predicted Sales Volume': original_forecast
})

print("\n==============================")
print("BMW SALES FORECAST")
print("==============================")

print(forecast_table)


# ============================================================
# 20. PLOT ORIGINAL SALES + FUTURE FORECAST
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    data_yearly.index,
    data_yearly.values,
    marker='o',
    label='Actual BMW Sales'
)

plt.plot(
    final_predictions.index,
    original_forecast,
    marker='o',
    label='Predicted BMW Sales'
)

plt.title('BMW Sales Forecast')
plt.xlabel('Year')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 21. SAVE FORECAST
# ============================================================

forecast_table.to_csv(
    'BMW_sales_forecast.csv',
    index=False
)

print("\nForecast saved as BMW_sales_forecast.csv")
```

<img width="1132" height="755" alt="image" src="https://github.com/user-attachments/assets/e4486471-edcf-49c3-9910-d2b5c84c123b" />
<img width="1000" height="457" alt="image" src="https://github.com/user-attachments/assets/9250f3fa-f3fe-4788-a9bb-d2f94cfb47a8" />
<img width="745" height="480" alt="image" src="https://github.com/user-attachments/assets/d2b230e7-e3ca-4c1f-93b7-f7ae28a8b619" />
<img width="1047" height="545" alt="image" src="https://github.com/user-attachments/assets/f5959cf8-e861-4f96-80ec-0c37496b5f16" />
<img width="1005" height="548" alt="image" src="https://github.com/user-attachments/assets/9ce9bef5-f1ad-4473-873d-c025c1ff81d1" />
<img width="1025" height="550" alt="image" src="https://github.com/user-attachments/assets/a2b368e1-f981-4496-880e-806360e9a142" />


# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
