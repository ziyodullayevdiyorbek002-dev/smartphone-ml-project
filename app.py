from flask import Flask
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Smartphone Addiction ML Model is Running on Azure!"

if __name__ == '__main__':
    app.run()