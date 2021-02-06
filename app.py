import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = int(request.form['Pregnancies'])
    Glucose = int(request.form['Glucose'])
    Insulin = int(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = int(request.form['Age'])
    
    test=[]
    test.append(Pregnancies)
    test.append(Glucose)
    test.append(Insulin)
    test.append(BMI)
    test.append(DiabetesPedigreeFunction)
    test.append(Age)
    
    test = [np.array(test)]
    result = model.predict(test)
    
    return render_template('index.html', prediction=result)
    
if __name__ == '__main__':
    app.run(debug=True)