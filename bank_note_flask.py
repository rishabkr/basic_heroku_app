# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:02:43 2020

@author: risha
"""
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle



app=Flask(__name__,template_folder='template') #mandatory first step
pickle_in=open('classfier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/') #root page
def home():
    return render_template('webpage.html')


@app.route('/predict',methods=['POST'])
def predict_note_auth():
    features=[int(x) for x in request.form.values()]
    prediction=classifier.predict([features])
    print(prediction)
    valid_dict={'0':'fake' ,'1':'authentic'}
    return render_template('webpage.html',prediction_text=f'The note is {valid_dict[str(prediction[0])]} !! ')

# @app.route('/predict_file',methods=['POST'])
# def predict_note_auth_file():
#     df_test=pd.read_csv(request.files.get('file'))
#     prediction=classifier.predict(df_test)
#     return f'The predicted values for the file is {prediction}'
#     variance=request.args.get('variance')
#     skewness=request.args.get('skewness')
#     curtosis=request.args.get('curtosis')
#     entropy=request.args.get('entropy')
#     predction=classifier.predict([[variance,skewness,curtosis,entropy]])
#     return f'prediction is {predction}'




    
if __name__=='__main__':
    app.run(debug=True)