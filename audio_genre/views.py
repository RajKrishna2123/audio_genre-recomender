from . import urls
from django.shortcuts import render
import os 
import requests
import json
import random
import tensorflow as tf
import subprocess
import librosa
from librosa.feature import mfcc
from sklearn.preprocessing import LabelEncoder
import numpy as np


lbenc = LabelEncoder()

# Define the command to be executed
command = "nvidia-smi"

# Execute the command and capture the output
output = subprocess.check_output(command, shell=True)

# Decode the output from bytes to string
output_str = output.decode("utf-8")

# Print the output
print(output_str)

# Load your saved model
print("loading model.......\n\n")
loaded_model = tf.keras.models.load_model('my_model.h5')
print("loading finished \n\n")



def home(request):
    return  render(request ,"index.html")
def search(request):
    q=request.GET.get('querry','default') 
    API_KEY = 'f94b77af55cac2a9c58533aa220884bc'
    API_SECRET = 'f4f698a08258776285f218546b2140ac'
    BASE_URL = "http://ws.audioscrobbler.com/2.0/"

    # Make the API request to get the top tracks for the input genre
    params = {
        "method": "tag.gettoptracks",
        "tag": q,
        "api_key": API_KEY,
        "format": "json"
    }
    response = requests.get(BASE_URL, params=params)

    # Parse the JSON response
    data = json.loads(response.text)
    dict={}
    j=0

    #loading complete data in structured manner 
    for i in data['tracks']['track']:
        j=j+1
        dict[j]=i

    #PICKING UP RANDOM 10 FROM THE PROVIDED LIST
    rand_10={}
    for i in range(1,10):
        x=random.randint(1,len(dict))
        rand_10[i]=dict[x]


    return  render(request ,"index.html",{'rand_10':rand_10})

def predict(request):
    return render(request, "predict.html")

def preprocess_audio(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path)
    # Get the MFCC features for the audio
    mfccs = mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512, n_fft=2048)
    # Reshape the MFCC features
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], 1))
    # Return the preprocessed data
    return mfccs

# Handle the uploaded audio file in a Django view
def handle_audio_upload(request):    
    audio_file = request.FILES['audio_file']
    save_dir = 'temp/audio'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, audio_file.name), 'wb+') as destination:
        for chunk in audio_file.chunks():
            destination.write(chunk)
        
    demo_path = os.path.join(save_dir,audio_file.name)
    processed_data = preprocess_audio(demo_path)
        
    predicted_probabilities = loaded_model.predict(processed_data)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)

    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    li=[]
    for pred_class in predicted_classes:
        li.append(classes[pred_class])
    print(li)
    return render(request ,"output.html" ,{"rawdata":li})


def get_frame_mfccs(path):
    audio, sr = librosa.load(path)
    frames = librosa.util.frame(audio, frame_length=sr*3, hop_length=sr*3)
    frame_mfccs = []
    for i in range(frames.shape[1]):
        mfccs = mfcc(y=frames[:,i],sr=sr,n_mfcc=13,hop_length=512,n_fft=2048)
        frame_mfccs.append(mfccs)
    return frame_mfccs

def reshape(data,shape=(26,65)):
    assert data.shape == (13,130) , f"The Data shape should be (13,130) but got {data.shape}"
    data = data.reshape(shape)
    data = np.expand_dims(data,axis=-1)
    return data

def preprocess_audio(path):
    frames_mfc = get_frame_mfccs(path)
    processed_data = np.array([reshape(x) for x in frames_mfc])
    processed_data = processed_data.astype(np.float32)
    return processed_data

def chat(request):
    import openai
    openai.api_key = "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"
    # create a completion
    completion = openai.Completion.create(model="ada", prompt="hi how are you")

    # print the completion
    print(completion.choices[0].text)
    raw={'bot':completion.choices[0].text,'user':""}
    return render(request , "chatbot.html",{"raw":raw})

def openai(request):
    q=request.POST.get('question','default') 
    import openai
    openai.api_key = "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"
    # create a completion
    completion = openai.Completion.create(model="ada", prompt=q)

    # print the completion
    print(completion.choices[0].text)
    raw={'bot':completion.choices[0].text,'user':q}
    return render (request, "chatbot.html",{"raw":raw})