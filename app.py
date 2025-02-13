from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from the pickle file
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        size = int(request.form['size'])
        rooms = int(request.form['rooms'])
        location = int(request.form['location'])
        
        # Predict the house price using the model
        prediction = model.predict(np.array([[size, rooms, location]]))
        
        # Prepare the result for displaying in a table
        result = {
            'Size (sqft)': size,
            'Rooms': rooms,
            'City': location,  # Display the encoded city number
            'Predicted Price': f"${prediction[0]:,.2f}"
        }
        
        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
