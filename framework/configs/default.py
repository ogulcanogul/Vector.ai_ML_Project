framework = {

    'model': {
        'structure': {
            'backbone': 'FashionResNet',
            'num_models': 1
        }
    },

    'optimizer': {
        'learning_rate': 1e-5,
        'gradient_transformers': {
            'weight_decay': 1e-4,
            'clipnorm': 10.0,
            'clipvalue': 10.0,
            'excluded_vars': ['bn', 'batchnorm']
        },
        'lr_multiplier': {},
        'method': 'Adam',
        'parameters': {} # default parameters
    },

    'training': {
        'max_epochs': 100,
        'early_stopping_patience': 15,
        'min_improvement_margin': 1e-5,
        'batch_size': 128,
        'preprocessing': 'FashionMNISTPreprocessing'
    },


    'validation': {
        'batch_size': 128,
        'preprocessing': 'FashionMNISTPreprocessing'
    },

    'test':{
        'batch_size': 128,
        'preprocessing': 'FashionMNISTPreprocessing'
    }

 }

backbone_models = {

    'FashionResNet': {
        'model_parameters': {
            'stages': 3,
            'blocks': 2,
            'block_repeats': 1,
            'dropout_probability': 0.25,
            'final_activation': True,
            'block_parameters': {
                'spatial_aggregation': 'skip',
                'pre_activation': True,
                'pre_activation_block': 'BN_ReLU',
                'mid_activation': True,
                'mid_activation_block': 'BN_ReLU',
                'post_activation': True,
                'post_activation_block': 'BN_ReLU',
                'input_injection': False,
                'zero_mean_embedding_kernel': False
            }

        },
        'input_parameters': {}

    }
}

optimizers = {

    'SGD': {'momentum': 0.0,
            'nesterov': False
            },

    'RMSprop': {
        'rho': 0.9,
        'momentum': 0.0
    },

    'Adam': {
        'beta_1': 0.9,
        'beta_2': 0.999
    },

}

class_labels = {  0: "T-shirt/Top",
  1: "Trouser",
  2: "Pullover",
  3: "Dress",
  4: "Coat",
  5: "Sandal",
  6: "Shirt",
  7: "Sneaker",
  8: "Bag",
  9: "Ankle Boot"}

