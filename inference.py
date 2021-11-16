import tensorflow as tf
import configparser
from framework.configs import getConfig
from framework.utilities import model_utils
from PIL import Image
from numpy import asarray
import numpy as np

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
file_path = './testData/03_00931_01.png'
test_data = Image.open(file_path)
test_data = (asarray(test_data))
test_data = np.expand_dims(test_data, axis=0)
test_data = tf.image.convert_image_dtype(test_data, dtype=tf.float32)
img_size = eval(config_params['backbone_models']['input_parameters']['img_size'])
test_data = tf.image.resize(test_data, img_size)
test_class = model_utils.inferenceModel(model_name, paths, test_data)
test_class = test_class[0].numpy()
print('The class of the test image is', config_params['class_labels'][test_class])


