import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word

def prog(inp):    
    data = pd.DataFrame(inp)
    
    # Doing some preprocessing on these tweets as done before
    data[0] = data[0].str.replace('[^\w\s]',' ')
    
    # From nltk.corpus import stopwords
    stop = stopwords.words('english')
    data[0] = data[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    # From textblob import Word
    data[0] = data[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    # Extracting Count Vectors Parameters using the vector pickel generated in Text-Emotion-Detection
    vectorizer = pickle.load(open("vector.pickel", "rb"))
    model_count = vectorizer.transform(data[0])
    
    #Predicting the emotion of the tweet using our already trained linear SVM
    model = pickle.load(open('model.pickel', 'rb'))
    model_pred = model.predict(model_count)
    
    if(model_pred[0] == 0):
        print("Emotion: Happiness")
    else:
        print("Emotion: Sadness")

# obtain audio from the microphone
def func():
    r = sr.Recognizer() 
       
    arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
         's','t','u','v','w','x','y','z']
    with sr.Microphone() as source:

        r.adjust_for_ambient_noise(source) 
        i=0
        while True:
            print('Say something')           

            # recognize speech using Uberi
            try:
                audio = r.record(source, duration = 5)
                a = r.recognize_google(audio)
                
                for c in string.punctuation:
                    a = a.replace(c,"")  
                                    
                print("you said " + a.lower())
                
                inp = []
                text = a.lower()
                inp.append(text) 
                prog(inp)
                
                for i in range(len(a)):
                    if(a[i] in arr):                
                        ImageAddress = 'letters/'+a[i]+'.jpg'
                        ImageItself = Image.open(ImageAddress)
                        ImageNumpyFormat = np.asarray(ImageItself)
                        plt.imshow(ImageNumpyFormat)
                        plt.draw()
                        plt.pause(0.8) # pause how many seconds
                    else:
                        continue
                
                plt.close()
                break
            except:
                print('Exception Occurred')

#func()
while 1:
    image = "signlang.png"
    msg = "HEARING IMPAIRMENT ASSISTANT"
    choices = ["Live Voice","All Done!"]
    reply = buttonbox(msg,image=image,choices=choices)
    if reply ==choices[0]:
        func()
    if reply == choices[1]:
        quit()