import base64
import uuid
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import matplotlib.pyplot as plt
from flask import Flask,render_template,redirect,url_for,request
import urllib3
from tensorflow.keras.preprocessing import image
import os



app = Flask(__name__)

model = load_model('shape2.h5')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif', 'PNG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict_(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    prediction = model.predict(img_tensor)

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction' , methods = ['GET' , 'POST'])
def predictions():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib3.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                predictions = predict_(img_path)

                if round(predictions[0][0]) == 0:
                    pred = 'No mask detected'
                else:
                    pred = 'Mask detected'
                
                predictions = {
                       'Prediction': pred
                }
                

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('prediction.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                predictions = predict_(img_path)

                if round(predictions[0][0]) == 0:
                    pred = 'No mask detected'
                else:
                    pred = 'Mask detected'
                
                predictions = {
                       'Prediction': pred
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('prediction.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
# new_image = load_image(img,show=True)
# pred = model.predict(new_image)