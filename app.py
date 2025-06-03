from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/disease_model.h5')
class_names = ['blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro', ...]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img = Image.open(image_file)
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)
            predicted_class = class_names[np.argmax(predictions[0])]
            return render_template('index.html', prediction=predicted_class)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)