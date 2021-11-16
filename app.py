from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import configparser
from framework.configs import getConfig
from framework.utilities import model_utils
from PIL import Image
from numpy import asarray
import os
import shutil

dataset = 'Fashion_MNIST'
path_to_config_file = './configs/config.ini'
verbose = 2
save_model = True

# paths to resources
paths = configparser.ConfigParser()
paths.read(path_to_config_file)

# parameters config
path_to_config_parameters = paths['DEFAULT']['CONFIG_PATH']

config_params = getConfig(
    config_dict={'base': 'config.yaml', 'dataset': dataset},
    configs_path=path_to_config_parameters)

model_name = 'FashionResNet'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = paths['DEFAULT']['UPLOAD_PATH']

# delete uploaded images
for filename in os.listdir(app.config['UPLOAD_FOLDER'] ):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'] , filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route('/')
def upload_f():
    return render_template('upload.html')


def predict(file_path):

    test_data = Image.open(file_path)
    test_data = (asarray(test_data))
    test_data = np.expand_dims(test_data, axis=0)
    test_data = tf.image.convert_image_dtype(test_data, dtype=tf.float32)
    img_size = eval(config_params['backbone_models']['input_parameters']['img_size'])
    test_data = tf.image.resize(test_data, img_size)
    test_class = model_utils.inferenceModel(model_name, paths, test_data)
    test_class = test_class[0].numpy()
    return str(config_params['class_labels'][test_class])


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        val = predict(file_path)
        return render_template('pred.html', ss=val)


if __name__ == '__main__':
    app.run()
