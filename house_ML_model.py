import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Sample data for House Price Prediction (e.g., size of house, number of rooms)
data = {
    'Size': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'Rooms': [2, 3, 3, 4, 4, 5, 5, 6, 6],
    'Location': [1, 2, 2, 3, 3, 4, 4, 5, 5],  # Encoded location (e.g., 1 = City A, 2 = City B)
    'Price': [150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000]
}

df = pd.DataFrame(data)

x = df[['Size', 'Rooms', 'Location']]
y = df['Price']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)