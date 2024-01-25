from flask import Flask, render_template, request
from chatbot import predict_class,get_response
import json

def apprender():
    intents = json.loads(open('./Intents.json',encoding='utf-8').read())

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template('botchat.html')


    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        ints = predict_class(msg)
        return get_response(ints, intents)
    return app



if __name__ == '__main__':
    app = apprender()
    app.run()(debug=True)