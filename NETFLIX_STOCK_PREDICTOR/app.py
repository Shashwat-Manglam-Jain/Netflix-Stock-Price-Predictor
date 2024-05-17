import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Initialize the scaler and load the trained model
sclr = StandardScaler()
model = pickle.load(open('netflix_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extracting input data from the form
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low = float(request.form['Low'])
        Adj_Close = float(request.form['Adj_Close'])
        Volume = int(request.form['Volume'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        # Creating the features array
        features = np.array([[Open, High, Low, Adj_Close, Volume, year, month, day]])
        
        # Scaling the Volume feature
        features[:, 4:] = sclr.fit_transform(features[:, 4:])
        
        # Making prediction
        prediction = model.predict(features)

        # Returning the result to the HTML template
        return render_template('index.html', output=prediction[0])
    except Exception as e:
        return str(e)

# Main function to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
