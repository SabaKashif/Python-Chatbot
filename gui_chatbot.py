#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model= load_model('chatbot_model.h5')
import json
import random


# In[2]:


intents_file = open('intents.json').read()
intents = json.loads(intents_file)
words=pickle.load(open('words.pkl', 'rb'))
classes= pickle.load(open('classes.pkl', 'rb'))
def clean_up_sentence(sentence):
    #tokenize the pattern- splitting words into array
    #sentence_words=nltk.word_tockenize(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    #stemming every word-reducing to base from sentence words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence


# In[3]:


def bag_of_words(sentence, words, show_details=True):
    #tockenize patterns
    sentence_words=clean_up_sentence(sentence)
    #bag of words- vocabulary matrix
    bag= [0] *len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                #assign 1 if current word is in the vocabulary position
                bag[i]=1
                if show_details:
                    print("found in bag: %s" %word)
        return(np.array(bag))


# In[4]:


def predict_class(sentence):
    #filter below threshold predictions
    p= bag_of_words(sentence, words, show_details=False)
    res=model.predict(np.array([p]))[0] 
    ERROR_THRESHOLD= 0.025
    results=[[i, r] for i,r in enumerate(res) if ERROR_THRESHOLD]
    #storing strength probability
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# In[5]:


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


# In[6]:


#creating tkinter GUI
import tkinter
from tkinter import *
def send():
    msg= EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "you:" +msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
        ints= predict_class(msg)
        res= getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
root= Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)


# In[7]:


#create chatbot window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatBox.config(state= DISABLED)


# In[8]:


#Bind scrollbar to chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set


# In[9]:


#Create button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )


# In[10]:


#create the box to enter message
EntryBox=Text(root, bd=0, bg="white", width="29", height="5", font="Arial")


# In[ ]:


#place all components on screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
root.mainloop()


# In[ ]:




