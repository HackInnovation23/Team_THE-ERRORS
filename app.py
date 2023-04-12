import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def result():
    return render_template('nextnew.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output==1:
        return render_template('nextnew.html', prediction_text='Congratulations! The candidate is placed.')
    else:
        return render_template('nextnew.html', prediction_text='Sorry! The candidate is not placed.')


if __name__ == '__main__':
    app.run("localhost","9999",debug=True)
