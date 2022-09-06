from flask import Flask, render_template, request, url_for, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
#from basis import lin_model



app = Flask(__name__)



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    with open("linear_regression.pkl", "rb") as file:
        model = pickle.load(file)
    Year = int(request.form['year'])
    Year = 2020-Year
    Bedroom = int(request.form['bedroom'])
    Bathroom = int(request.form['bathroom'])
    Kitchen = int(request.form['kitchen'])

    Sale_condition = request.form['sale']
    if(Sale_condition=='normal'):
        SaleCondition_Normal=1
        SaleCondition_Abnorml=0

    elif(Sale_condition=='abnormal'):
        SaleCondition_Normal=0
        SaleCondition_Abnorml=1


    arr = [Year,Bedroom,Bathroom,Kitchen, SaleCondition_Abnorml,SaleCondition_Normal]
    clean_data = [int(i) for i in arr]
    ex1 = np.array(clean_data).reshape(1,-1)

    prediction = model.predict(ex1)
    output=round(prediction[0],2)
    output = output * 360
    if output<0:
        return render_template('result.html',prediction="Sorry You Cannot Sell This House")
    else:
        return render_template('result.html',prediction="You Can Sell The House For N{}".format(output))



if __name__== "__main__":
    app.run(debug=True)
