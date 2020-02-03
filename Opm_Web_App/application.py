# -*- coding: utf-8 -*-
"""
OPM Web Application Template
"""

import json
import plotly
from flask import Flask
from flask import render_template, request, redirect, url_for
from plotly.graph_objs import Bar

from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import pandas as pd
import os
import itertools


import boto3
from smart_open import smart_open

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = '' 


def normalize_app(df):
    
    '''transform input dataframe and create numpy array'''
    
    df = df.groupby('EFDATET').sum()
    
    for col in list(df.columns):
       
        mean, std = df[col].mean(), df[col].std()

        df.loc[:, col] = (df[col] -mean) /(std + 1) 
    
    features = df.reset_index().iloc[:,1:]
    
    features = features.values
        
    return features 


def plot_pred_app(preds):
    
    '''transform scaled outputs of model into actual outputs'''
    
    # estimated statisitical parameters of output label
    
    train_label_mean = 4114.48
    
    train_label_std = 1422.75
    
    unnormal_preds = preds * (train_label_std + 1) + train_label_mean
    
    merged = list(itertools.chain(*unnormal_preds))
    
    return merged 


application = Flask(__name__, static_url_path='')

upload_path = '/static/upload'


@application.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
       
        file = request.files['file'] 
                
        s3 = boto3.resource('s3', 
                     aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name='us-east-1'
                      )
                 
        object = s3.Object('elasticbeanstalk-us-east-1-357561279081', 'wyoming/upload/df_input.csv')
        
        object.put(Body=file)       
              
        return redirect(url_for('go'))
    else:
        return render_template('master.html')
    

@application.route('/go', methods=['GET', 'POST'])
def go():
    
    '''load model and recompile using provided weights to produce voluntary separation 
    estimates by time period rendered as a bar graph on another page'''
    
    aws_key=AWS_ACCESS_KEY_ID
    aws_secret=AWS_SECRET_ACCESS_KEY
    bucket_name = 'elasticbeanstalk-us-east-1-357561279081'
    object_key = 'wyoming/upload/df_input.csv'

    path = 's3://{}:{}@{}/{}'.format(aws_key, aws_secret, bucket_name, object_key)

    web_input = pd.read_csv(smart_open(path))

    app_features = normalize_app(web_input.iloc[:,1:])
    
    model_file = os.path.join('static/model/', 'model.json')
    
    json_file = open(model_file, 'r')
    
    K.clear_session()
    
    loaded_model_json = json_file.read()
    
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)    
    
    loaded_model.load_weights(os.path.join('static/weights/','best.h5'))
    
    sgd = optimizers.SGD(lr=0.01)
    
    loaded_model.compile(loss=losses.mean_squared_error,
                  optimizer=sgd,
                  metrics=['mean_squared_error'])    
        
    results = loaded_model.predict(app_features)
    
    results = plot_pred_app(results)
    
    period = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    bar_chart = Bar(x = period, y = results)
    
    bar_json = json.dumps(bar_chart, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(bar_json)
     
    return render_template('go.html', bar_json = bar_json) 


    
if __name__ == "__main__":
    application.run(debug=False)
