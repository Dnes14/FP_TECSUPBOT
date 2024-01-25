from flask import Flask, render_template, request, jsonify
from chatbot import predict_class,get_response
import json

intents = json.loads(open('intents.json',encoding='utf-8').read())

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('botchat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    ints = predict_class(msg)
    return get_response(ints, intents)



if __name__ == '__main__':
    app.run()