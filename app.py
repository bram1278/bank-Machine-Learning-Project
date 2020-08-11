from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/visual')
def visual():
    df = pd.read_csv('bank.csv')


    # Menentukan Size
    x = df['CreditScore']
    fig = plt.hist(x, 10, rwidth=0.9)
    plt.title("Histogram Credit Score Nasabah")
    plt.xlabel("Credit Score Nasabah")
    plt.ylabel("Jumlah Freq")
    
    plt.savefig('creditscore_hist.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    x = df['Balance']   
    fig = plt.hist(x, 10, rwidth=0.9)
    plt.title("Histogram Deposito (Balance) Nasabah")
    plt.xlabel("Deposito (Balance) Nasabah")
    plt.ylabel("Jumlah Freq")


    plt.savefig('balance_hist.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result2 = str(figdata_png)[2:-1]

    return render_template('plot.html', plot=result, plot2= result2 )




@app.route('/bank')
def bank():
    return render_template('bank.html')

@app.route('/klasifikasi', methods = ['POST', 'GET'])
def hasil():
    if request.method == 'POST':
        input = request.form
        age = float(input['age'])
        balance = float(input['balance'])
        active = float(input['active'])
        france = float(input['france'])
        germany = float(input['germany'])
        spain = float(input['spain'])
        female = float(input['female'])
        male = float(input['male'])
        pred = Model.predict([[age,balance,active,france,germany,spain,female,male]])[0]

        return render_template('hasil.html', data=input, prediksi=pred)

if __name__ == "__main__":
    with open('bankModel', 'rb') as model:
        Model = pickle.load(model)
    app.run(debug=True)













