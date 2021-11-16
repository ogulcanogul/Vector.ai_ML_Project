from . import default

import yaml
import os

def updateConfig(base_config: dict, ref_config: dict):

    for k, v in ref_config.items():

        if k in base_config.keys():

            if isinstance(v, dict):
                updateConfig(base_config[k], v)
            else:
                base_config[k] = v
        else:
            base_config.update({k: v})

    return base_config

def updateOptimizerConfig(base_config: dict):

    method = base_config['optimizer']['method']
    if base_config['optimizer']['parameters'] == {}:
        base_config['optimizer']['parameters'] = \
            default.optimizers[method]

    if method.lower().endswith('lrm'):
        base_config['optimizer']['parameters']['lr_multiplier'].update(base_config['optimizer'].pop('lr_multiplier'))


def mergeConfigs(config_dict, configs_path=''):

    if not isinstance(config_dict, dict):
        with open(os.path.join(configs_path, config_dict), 'r') as stream:
            config_dict = yaml.safe_load(stream)


    if 'base' in config_dict.keys():
        base_config = mergeConfigs(config_dict.pop('base'), configs_path)

    else:
        base_config = default.framework


    return updateConfig(base_config, config_dict)

def getConfig(config_dict, configs_path=''):

    config = mergeConfigs(config_dict, configs_path)

    updateOptimizerConfig(config)

    return config







