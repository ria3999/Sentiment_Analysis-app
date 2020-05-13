from flask import Flask,render_template,redirect,session,flash,request,url_for
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf

from numpy import array
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


IMAGE_FOLDER=os.path.join("static","img_pool")


app1=Flask(__name__)
app1.config['UPLOAD FOLDER']=IMAGE_FOLDER


model=load_model("final1_model.h5")
   


@app1.route('/',methods=['GET','POST'])
def home():
    return render_template("ria.html")

@app1.route('/result',methods=['POST','GET'])
def result():
    if request.method=='POST':
        text=request.form['text']
        sentiment=''
        max_review_length=500
        word_to_id=imdb.get_word_index()
        strip_specialchars=re.compile("[^A-Za-z0-9]+")
        text=text.lower().replace("<br />"," ")
        text=re.sub(strip_specialchars," ",text.lower())
        words=text.split()
        x_test=[[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000)else 0 for word in words]]
        x_test=sequence.pad_sequences(x_test,maxlen=500)
        vector=np.array([x_test.flatten()])
        probability=model.predict(array([vector][0]))[0][0]
        class1=model.predict_classes(array([vector][0]))[0][0]
        if class1==0:
            sentiment="Negative"
            img_filename="https://i.imgur.com/bvFmcV5.png"
        else:
            sentiment="Positive"
            img_filename="https://i.imgur.com/KfezKO5.jpg"
    return render_template('ria.html',text=text,sentiment=sentiment,probability=probability,image=img_filename)

if __name__=="__main__":
    app1.run()
