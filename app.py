from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
from numpy.random import randint, choice
import pickle
import math

result = {'PHASE 1': ['Breathing Exercise', 'Breathing Exercise', 'Breathing Exercise', 'Breathing Exercise', 'Breathing Exercise'],
 'PHASE 2': ['Light Exercise', 'Light Exercise', 'Light Exercise', 'Light Exercise', 'Light Exercise'],
 'PHASE 3': ['Everyday Activities (Balance while Sitting)', 'Everyday Activities (Balance while Sitting)', 'Everyday Activities (Balance while Sitting)', 'Everyday Activities (Balance while Sitting)', 'Everyday Activities (Balance while Sitting)'],
 'PHASE 4': ['Everyday Activities (Exercises while Sitting)', 'Everyday Activities (Give Support to your Weak Side)', 'Everyday Activities (Muscles Lengthening hand, arm and leg )', 'Everyday Activities (Exercises while Sitting)', 'Everyday Activities (Give Support to your Weak Side)'],
 'PHASE 5': ['Everyday Activities (Muscles Lengthening hand, arm and leg )', 'Everyday Activities (Muscles Lengthening hand, arm and leg )', 'Everyday Activities (Exercises while Sitting)', 'Everyday Activities (Muscles Lengthening hand, arm and leg )', 'Everyday Activities (Muscles Lengthening hand, arm and leg )'],
 'PHASE 6': ['Everyday Activities (Give Support to your Weak Side)', 'Everyday Activities (Strengthening your Arm, Hand and Legs)', 'Everyday Activities (Strengthening your Arm, Hand and Legs)', 'Everyday Activities (Give Support to your Weak Side)', 'Everyday Activities (Strengthening your Arm, Hand and Legs)'],
 'PHASE 7': ['Everyday Activities (Strengthening your Arm, Hand and Legs)', 'Everyday Activities (Exercises while Sitting)', 'Everyday Activities (Give Support to your Weak Side)', 'Everyday Activities (Strengthening your Arm, Hand and Legs)', 'Everyday Activities (Exercises while Sitting)'],
 'PHASE 8': ['Everyday Activities (Functional Hand and Finger Activities)', 'Everyday Activities (Functional Hand and Finger Activities)', 'Everyday Activities (Functional Hand and Finger Activities)', 'Everyday Activities (Functional Hand and Finger Activities)', 'Everyday Activities (Functional Hand and Finger Activities)'],
 'PHASE 9': ['Everyday Activities (Stand up and Getting up from fall)', 'Everyday Activities (Bed moving, transferring and rolling)', 'Everyday Activities (Bed moving, transferring and rolling)', 'Everyday Activities (Stand up and Getting up from fall)', 'Everyday Activities (Stand up and Getting up from fall)'],
 'PHASE 10':['Everyday Activities (Bed moving, transferring and rolling)', 'Everyday Activities (Stand up and Getting up from fall)', 'Everyday Activities (Stand up and Getting up from fall)', 'Everyday Activities (Bed moving, transferring and rolling)', 'Everyday Activities (Bed moving, transferring and rolling)'],
 'PHASE 11':['Standing and Walking (Exercise for a very weak leg -wearing brace or gaiter)', 'Standing and Walking (Exercise for a very weak leg -wearing brace or gaiter)', 'Standing and Walking (Exercise for a very weak leg -wearing brace or gaiter)', 'Standing and Walking (Exercise for a very weak leg -wearing brace or gaiter)', 'Standing and Walking (Exercise for a very weak leg -wearing brace or gaiter)'],
 'PHASE 12':['Standing and Walking (exercises without a knee support)', 'Standing and Walking (standing)', 'Standing and Walking (exercises without a knee support)', 'Standing and Walking (exercises without a knee support)', 'Standing and Walking (standing)'],
 'PHASE 13':['Standing and Walking (walking)', 'Standing and Walking (exercises without a knee support)', 'Standing and Walking (exercises without a knee support)', 'Standing and Walking (walking)', 'Standing and Walking (exercises without a knee support)'],
 'PHASE 14':['Strength and Control (Movement in your Arm and Hand)', 'Standing and Walking (walking)', 'Strength and Control (Movement in your Arm and Hand)', 'Strength and Control (Movement in your Arm and Hand)', 'Standing and Walking (walking)'],
 'PHASE 15':['Strength and Control (Strengthen your Back and Stomach Muscles)', 'Strength and Control (Movement in your Arm and Hand)', 'Strength and Control (Strengthen your Back and Stomach Muscles)', 'Strength and Control (Strengthen your Back and Stomach Muscles)', 'Strength and Control (Movement in your Arm and Hand)'],
 'PHASE 16':['Strength and Control (Strengthen  Hip)', 'Strength and Control (Strengthen your Back and Stomach Muscles)', 'Strength and Control (Strengthen  Hip)', 'Strength and Control (Strengthen  Hip)', 'Strength and Control (Strengthen your Back and Stomach Muscles)'],
 'PHASE 17':['Strength and Control (Strengthen Ankle)', 'Strength and Control (Strengthen  Hip)', 'Strength and Control (Strengthen Ankle)', 'Strength and Control (Strengthen Ankle)', 'Strength and Control (Strengthen  Hip)'],
 'PHASE 18':['Strength and Control (Strengthen  Knee)', 'Strength and Control (Strengthen Ankle)', 'Strength and Control (Strengthen  Knee)', 'Strength and Control (Strengthen  Knee)', 'Strength and Control (Strengthen Ankle)'],
 'PHASE 19':['Facial Exercise (Smiling and talking)', 'Strength and Control (Strengthen  Knee)', 'Facial Exercise (Smiling and talking)', 'Facial Exercise (Smiling and talking)', 'Strength and Control (Strengthen  Knee)'],
 'PHASE 20':['Speech Training Exercises (Lip Buzing , Vowal and consonant sounds exercises)', 'Facial Exercise (Smiling and talking)', 'Speech Training Exercises (Lip Buzing , Vowal and consonant sounds exercises)', 'Speech Training Exercises (Lip Buzing , Vowal and consonant sounds exercises)', 'Facial Exercise (Smiling and talking)'],
 'PHASE 21':['Return to work assistant', 'Speech Training Exercises (Lip Buzing , Vowal and consonant sounds exercises)', 'Return to work assistant', 'Return to work assistant', 'Speech Training Exercises (Lip Buzing , Vowal and consonant sounds exercises)'],
 'PHASE 22':['Rest', 'Return to work assistant', 'Rest', 'Rest', 'Return to work assistant']}


num_disease = []
app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Load the scaler
fp = open("scaler.bin", "rb")
scaler = pickle.load(fp)
fp.close()

# Load LinearSVM
fp = open("LinearSVM.bin", "rb")
model1 = pickle.load(fp)
fp.close()

# Load LogisticRegression
fp = open("LogisticRegression.bin", "rb")
model2 = pickle.load(fp)
fp.close()

@app.route("/result",methods=['POST','GET'])
def result():
    disease = []
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    if hypertension == 1:
        disease.append("HYPERTENSION")
        num_disease.append(3)
    heart_disease = int(request.form['heart_disease'])
    if heart_disease == 1:
        disease.append('HEART DISEASE')  
        num_disease.append(2)
        
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    if avg_glucose_level > 199:
        disease.append("DIABETES")
        num_disease.append(4)
    
    bmi = float(request.form['bmi'])
    if bmi > 30 and bmi < 35:
        disease.append("OBESITY I")
        num_disease.append(1)
    elif bmi > 35 and bmi < 40:
        disease.append("OBESITY II")
        num_disease.append(1)
    elif bmi > 40:
        disease.append("OBESITY III")
        num_disease.append(1)
    
    
    smoking_status = int(request.form['smoking_status'])
    if smoking_status == 3:
        disease.append("SMOKING")
        num_disease.append(0)
    

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('D:/Python37/Projects/Stroke Prediction','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('D:/Python37/Projects/Stroke Prediction','models/dt.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        pred=1
        return render_template('stroke.html',prediction_text=pred, dis=",".join(disease))
    
@app.route('/index',methods=['POST','GET'])
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[float(x) for x in request.form.values()]]
    final_features = scaler.transform(int_features)
    #output = model.predict(final_features)
    # Now convert linear svm's prediction to probability
    out1 = sigmoid(model1.decision_function(final_features))
    # Get prediction probability from logistic regression
    out2 = model2.predict_proba(final_features)[:,1]

    # Their average is the final output probability
    final_output = np.mean((out1, out2), axis=0)
    #output = prediction.reshape(-1, 1)
    output=(str(round(final_output[0]*100, 2)))
    return render_template('index.html', prediction_text='Severity of brain stroke {}%'.format(output))  
    
@app.route("/expert",methods=['POST','GET'])
def expert():
    return render_template('expert-review.html')

@app.route("/expert_review", methods=['POST','GET'])
def expert_review():
    PHASE,HEALTH,EXERCISE = "","",""
    FASE = request.form["phase"]
    PROGRESS =  int(request.form.get("proses", False))
    if PROGRESS > 70 :
        PHASE = FASE.split(' ')[0] +" "+str(int(FASE.split(" ")[1])+1)
    else:
        PHASE = FASE
    HEALTH = choice([str(x)+"%" for x in range(30,90,10)])
    EXERCISE = [result[PHASE][x] for x in list(set(num_disease))]
    return render_template('expert-review.html')

    
if __name__=="__main__":
    app.run(debug=True,port=7384)