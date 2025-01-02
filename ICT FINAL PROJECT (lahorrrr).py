import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values by filling with median
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Encode categorical columns like 'wd' and 'station'
    label_encoders = {}
    for col in ['wd', 'station']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Create a datetime column
    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data.set_index('datetime', inplace=True)

    # Drop non-relevant columns
    data.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True)

    return data, label_encoders

# Split data into features and target
def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Make predictions for the next 30 days with disparity in AQI values
def predict_next_30_days(model, data, start_date):
    future_dates = pd.date_range(start=start_date, periods=30, freq='D')
    future_data = pd.DataFrame(index=future_dates)

    # Fill future_data with random variations based on median values
    for col in data.columns:
        if col != 'PM2.5':  # Exclude target column from prediction dataset
            future_data[col] = data[col].median() + np.random.uniform(-5, 5, size=len(future_data))

    # Ensure the columns are in the same order as the training data
    future_data = future_data[data.drop(columns=['PM2.5']).columns]

    future_predictions = model.predict(future_data)

    return future_dates, future_predictions

# Categorize AQI levels
def categorize_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Display textual output
def textual_output(dates, predictions):
    print("\nPredicted AQI Levels for the Next 30 Days:\n")
    summary = {}
    for date, prediction in zip(dates, predictions):
        category = categorize_aqi(prediction)
        print(f"Date: {date.date()}, Predicted AQI: {prediction:.2f}, Category: {category}")

        # Count occurrences of each category
        if category not in summary:
            summary[category] = 0
        summary[category] += 1

    print("\nSummary of Predicted AQI Categories:\n")
    for category, count in summary.items():
        print(f"{category}: {count} days")

# Plot predictions
def plot_predictions(dates, predictions):
    plt.figure(figsize=(12, 8))
    plt.plot(dates, predictions, marker='o', linestyle='-', label='Predicted AQI')

    # Add AQI thresholds
    plt.axhline(y=50, color='green', linestyle='--', label='Good Threshold')
    plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate Threshold')
    plt.axhline(y=150, color='orange', linestyle='--', label='Unhealthy for Sensitive Groups')
    plt.axhline(y=200, color='red', linestyle='--', label='Unhealthy Threshold')
    plt.axhline(y=300, color='purple', linestyle='--', label='Very Unhealthy Threshold')

    plt.title('Predicted AQI Levels for the Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # File path to the dataset
    file_path = r'C:\Users\Susan\Downloads\Lahore.xlsx.csv'

    # Load and preprocess data
    data = load_data(file_path)
    data, label_encoders = preprocess_data(data)

    # Split into features and target (predicting PM2.5)
    X, y = split_data(data, target_column='PM2.5')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Predict for the next 30 days
    start_date = data.index.max() + pd.Timedelta(days=1)
    future_dates, future_predictions = predict_next_30_days(model, data, start_date)

    # Display textual output
    textual_output(future_dates, future_predictions)

    # Plot predictions
    plot_predictions(future_dates, future_predictions)

if __name__ == "__main__":
    main()
