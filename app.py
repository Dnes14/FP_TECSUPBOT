
from flask import Flask, render_template, request
from chatbot import predict_class,get_response
import json

# Ingreso 
intents = json.loads(open('./Intents.json',encoding='utf-8').read())

app = Flask(__name__)

#Proceso
    #Enrutamiento

@app.route("/")
def index():
    return render_template('botchat.html')


#Salida
    #Implemetacion del chat bot

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    ints = predict_class(msg)
    return get_response(ints, intents)

# Auxiliar de ejecuccion de flask

if __name__ == '__main__':
    app.run()(debug=True)