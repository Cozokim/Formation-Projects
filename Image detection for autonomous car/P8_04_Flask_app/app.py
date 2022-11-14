import flask
from flask import Flask, render_template, request
import os
import tensorflow as tf
from keras.models import load_model 
from keras.preprocessing import image
import numpy as np
import keras 
from keras.preprocessing.image import save_img, img_to_array

image_folder = os.path.join('static', 'images')


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = image_folder

model=tf.keras.models.load_model('./data/model/model')


@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  # predicting images
  imagefile = request.files['imagefile']
  image_path = './static/images/' + imagefile.filename 
  imagefile.save(image_path)

  img = image.load_img(image_path, target_size=(512, 512))
  img_size = (512, 512)
  img = np.empty((1, *img_size + (3,)))
  img[0,] = img[0][0][0]

  res = model.predict(img)


  def combineMask(masks):
    _output = np.empty((512,512) + (1,))
    _x, _y = 0, 0
    
    for x in range(0, masks.shape[0]):
        for y in range(0, masks.shape[1]):
                   _target = masks[x][y]
                   _output[x][y] = np.argmax(_target) 
    return _output

  testing = combineMask(res[0])


  mask = keras.preprocessing.image.array_to_img(
    testing, data_format=None, scale=True, dtype=None,
    )

  pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
  mask_ = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + imagefile.filename)

  output_path = './static/images/mask-' + imagefile.filename 
  save_img(output_path, img_to_array(mask))

   
  return render_template('index.html', user_image=pic,mask_image=mask_)
 
