import tensorflow as tf
import configparser
from framework.configs import getConfig
from framework.utilities import model_utils


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


score, model_name = model_utils.trainXValModel(paths, config_params, save_model=save_model, verbose=verbose)

test_score = model_utils.evaluateModel(model_name, paths, config_params, verbose=1)





