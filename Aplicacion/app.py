from flask import Flask
from flask import render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import base64
import io
from PIL import Image
import numpy as np
from flask import jsonify
import logging
from flask import send_from_directory
import cv2






UPLOAD_FOLDER = 'C:/Users/blanca/anaconda3/envs/tensorflow/upload_folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





@app.route('/')
def index():
    return render_template('index.html')


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
#model.load_weights('model.h5py')

def preprocesamiento_img(img):
    #cargar_img = load_img(img)
    img_array = img_to_array(img)
    img_reshape = img_array.reshape((1,) + img_array.shape)
    return img_reshape

label_dict={0:'Covid19 positivo', 1:'Opacidad pulmonar', 2: 'Covid19 negativo', 3:'Neumonia viral'}

@app.route("/", methods=['POST', 'GET'])
def predict():
    
    if request.method == 'POST':
        if request.files.get('file'):
            
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            #if img.size != (299,299)
            
            img = np.array(img)
            img = cv2.resize(img, (299, 299))
            img = img/255
            img = np.expand_dims(img, axis=0)
            
            prediction = model.predict(img)
            
            result = np.argmax(prediction,axis=1)[0]
            accuracy = float(np.max(prediction,axis=1)[0])
            label= label_dict[result]

            print(prediction,result,accuracy)
            response = {'prediction': {'result': label,'accuracy': accuracy}}
 
            return render_template('index.html', 
                                   prediction_text = "Con una precisión del {0} la imagen que usted ha seleccionado es {1}".format(response["prediction"]["accuracy"],response["prediction"]["result"]) )



if __name__ == "__main__":
    app.run(debug = True, port = '8090')
    
    
    
    
    
    
    
    
