from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
from flask_wtf import FlaskForm
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, SelectField, IntegerField
import joblib
import numpy as np



# Register the user with wtforms
class PatientForm(Form):
    
    name = StringField("Name Surname", validators=[validators.Length(min=4, max=25), 
                                                    validators.DataRequired()])
    feature_5 = StringField("Feature_5", validators=[validators.Length(min=1, max=25),
                                                    validators.DataRequired()])
    feature_27 = SelectField(u'Feature_27', choices=[('Yes', 'Yes'), ('No', 'No')])
    feature_32 = SelectField(u'Feature_32', choices=[('Yes', 'Yes'), ('No', 'No')])
    feature_37 = SelectField(u'Feature_37', choices=[('Yes', 'Yes'), ('No', 'No')])
    feature_39 = SelectField(u'Feature_39', choices=[('Yes', 'Yes'), ('No', 'No')])
    feature_40 = SelectField(u'Feature_40', choices=[('Yes', 'Yes'), ('No', 'No')])
    feature_43 = SelectField(u'Feature_43', choices=[('Yes', 'Yes'), ('No', 'No')])

app = Flask(__name__)
model = joblib.load('sdsp_model.sav')
scaler = joblib.load('sdsp_scaler.sav')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'



@app.route("/", methods=["GET"])
def index():

    form = PatientForm(request.form)
    return render_template("index.html", form=form)

@app.route("/predict", methods=["GET", "POST"])
def predict():	
    form = PatientForm(request.form)
    if request.method == "POST" and form.validate():
        
        features = [x for x in request.form.values()]
        name = features[0]
        feature_5 = float(features[1])
        yes_no_features = [1 if i=='Yes' else 0 for i in features[2:]]
        features = np.concatenate((feature_5, yes_no_features), axis=None).reshape(1, 7)
        targets = ['Disease_1', 'Disease_2', 'Disease_3', 'Disease_4']
        
        scaled_features = scaler.transform(features)
        preds = model.predict_proba(scaled_features)[0]
    
        disease_1 = "{} prediction is {:.2f}".format(targets[0], (float(preds[0])))
        disease_2 = "{} prediction is {:.2f}".format(targets[1], (float(preds[1])))
        disease_3 = "{} prediction is {:.2f}".format(targets[2], (float(preds[2])))
        disease_4 = "{} prediction is {:.2f}".format(targets[3], (float(preds[3])))
            
                    
    return render_template("result.html", PatientName=name, disease_1=disease_1, disease_2=disease_2, disease_3=disease_3, disease_4=disease_4)
	

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
