import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template # for creating application lightweight web app
import numpy as np
import pandas as pd

app=Flask(__name__) # starting point of the application from where it will run
## Load regression and scalar models
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/') # first root (home page)
def home(): # create home page
    return render_template('home.html') # return html page

@app.route('/predict_api',methods=['POST']) # create predictive in a form of API for testing on Postman (POST: put some input to the model)
def predict_api():
    data=request.json['data'] # when hit /predict_api -> post request data with json form
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # -1: don't care how many characters but dimension need to be same as 1st parameter
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST']) # create predictive in Front-End application
def predict():
    data=[float(x) for x in request.form.values()] # the value that were filled in that form will be captured and change to float every value inside request.form
    final_input=scalar.transform(np.array(data).reshape(1,-1)) 
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     
