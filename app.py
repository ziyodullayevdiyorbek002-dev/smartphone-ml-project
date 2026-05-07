from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    screen_time = float(request.form['screen_time'])
    social_media = float(request.form['social_media'])
    gaming = float(request.form['gaming'])
    work = float(request.form['work'])
    sleep = float(request.form['sleep'])
    notifications = int(request.form['notifications'])
    app_opens = int(request.form['app_opens'])

    features = np.array([[
        age,
        gender,
        screen_time,
        social_media,
        gaming,
        work,
        sleep,
        notifications,
        app_opens
    ]])

    prediction = model.predict(features)

    result = "User is Addicted" if prediction[0] == 1 else "User is Not Addicted"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)