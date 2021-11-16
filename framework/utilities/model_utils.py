import tensorflow as tf

from sklearn.decomposition import PCA

import os
import shutil
import pprint

from .. import datasets
from .. import models
from .. import layers
from .. import utilities
from .. import configs
from .. import solvers

def addRegularizationLoss(model, loss_fn):

    for layer in model.layers:

        if hasattr(layer, 'layers'):
            addRegularizationLoss(layer, loss_fn)

        if hasattr(layer, 'kernel'):
            model.add_loss(lambda layer=layer: loss_fn(layer.kernel))



def logTraining(model_name, score, training_path, params, **hyperparams):

    for key, val in hyperparams.items():
        subkeys = key.split('/')
        hyp = params
        for subkey in subkeys[:-1]:
            hyp = hyp[subkey]
        hyp[subkeys[-1]] = val

    optimizer_id = params['optimizer']['method']

    top_line = '\n=== training configuration ===\n'
    bottom_line = '=== end of training configuration ===\n'
    training_line = 'optimizer: {}, learning_rate: {}, weight_decay: {}\n'.format(
        optimizer_id,
        params['optimizer']['learning_rate'],
        params['optimizer']['gradient_transformers']['weight_decay'])

    score_line = 'score: {}\n'.format(score)

    fpath = os.path.join(training_path, '{}.txt'.format(model_name))
    with open(fpath, 'a') as f:
        f.writelines([top_line, training_line, score_line, bottom_line])

def trainModel(paths, config_params,  save_model=True, verbose=0, **hyperparams):

    params = config_params

    # set hyperparameters
    if verbose and len(hyperparams):
        print('Hyperparameters:')
    for key, val in hyperparams.items():
        subkeys = key.split('/')
        hyp = params
        for subkey in subkeys[:-1]:
            hyp = hyp[subkey]
        hyp[subkeys[-1]] = val
        if verbose:
            print('{}: {}'.format(key, hyp[subkeys[-1]]))


    dataset_id = params['dataset']
    optimizer_id = params['optimizer']['method']

    backbone = params['model']['structure']['backbone']
    # ================================================================================================


    model_identifier = backbone

    if verbose:
        print('Training Model: {}'.format(model_identifier))

    # ================================================================================================

    # dataset
    # ================================================================================================
    dataset = getattr(datasets, dataset_id)(dataset_dir=paths['DEFAULT']['DATASETS_PATH'], verbose=verbose)

    preprocessing_train = getattr(utilities.dataset_utils,
                            params['training']['preprocessing'])(
        **configs.default.backbone_models[backbone]['input_parameters'])

    preprocessing_val = getattr(utilities.dataset_utils,
                            params['validation']['preprocessing'])(
        **configs.default.backbone_models[backbone]['input_parameters'])
    val_set = 'val' if dataset.num_classes.val > 0 else 'eval'

    #print('\tpreparing dataset...')
    train_data = dataset.makeBatch(subset='train',
                                   preprocess_fn= preprocessing_train, batch_size=params['training']['batch_size'])

    val_data = dataset.makeBatch(subset= val_set,
                                 preprocess_fn= preprocessing_val,
                                 batch_size=params['validation']['batch_size'])

    #print('\tdataset is ready!')
    #print('\tbuilding model...')

    # model
    # ================================================================================================
    model = \
        getattr(models, backbone)(input_shape=list(preprocessing_train.out_image_size), num_classes= dataset.num_classes.train,
                                  **configs.default.backbone_models[backbone]['model_parameters'])

    gradient_transformers = []
    model_callbacks = []

    # loss and metric
    # ================================================================================================
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    metric_callback = tf.keras.metrics.SparseCategoricalAccuracy()

    # callbacks
    # ================================================================================================
    early_stopping_callback = \
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                         min_delta=params['training']['min_improvement_margin'],
                                         patience=params['training']['early_stopping_patience'],
                                         verbose=verbose,
                                         mode='max',
                                         restore_best_weights=True)


    logdir = os.path.join(paths['DEFAULT']['TRAINING_PATH'], 'logs', model_identifier)
    tensorboard_callback = \
        tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                       write_graph=False,
                                       histogram_freq=0,
                                       update_freq='epoch',
                                       profile_batch=0)

    # optimizer
    # ================================================================================================
    gradient_transformers += solvers.gradient_transformers.getGradientTransformers(
        model=model,
        **params['optimizer']['gradient_transformers'])
    optimizer_class = getattr(solvers.Optimizers, optimizer_id)

    if tf.__version__ >= '2.4':
        optimizer = optimizer_class(**params['optimizer']['parameters'],
                                    learning_rate=params['optimizer']['learning_rate'],
                                    gradient_transformers=gradient_transformers)
    else:
        optimizer = solvers.buildOptimizer(optimizer_class,
                                           **params['optimizer']['parameters'],
                                           learning_rate=params['optimizer']['learning_rate'],
                                           gradient_transformers=gradient_transformers)


    steps_per_epoch = dataset.size.train_split[0] // params['training']['batch_size']



    # Compile model by specifying the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=optimizer,
                  # Loss function to minimize
                  loss=loss,
                  # List of metrics to monitor
                  metrics=metric_callback)

    # actual long term training
    # ================================================================================================
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    file_writer = tf.summary.create_file_writer(logdir=logdir)
    file_writer.set_as_default()

    epochs = params['training']['max_epochs']

    train_history = model.fit(train_data, epochs=epochs, validation_data=val_data, steps_per_epoch=steps_per_epoch,
                              callbacks=[early_stopping_callback, tensorboard_callback] + model_callbacks,
                              verbose=bool(verbose))

    score = max(train_history.history['val_sparse_categorical_accuracy'])

    if save_model:
        training_dir = paths['DEFAULT']['TRAINING_PATH']
        save_file = os.path.join(training_dir, model_identifier + '.h5')
        model.save(save_file, overwrite=True, include_optimizer=False, save_format='h5')

    del model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return score, model_identifier

def trainXValModel(paths, config_params, save_model=True, verbose=0, **hyperparams):

    # ================================================================================================



    model_score, model_name = trainModel(paths=paths,
                                             config_params=config_params,
                                             save_model=save_model,
                                             verbose=verbose, **hyperparams)


    logTraining(model_name, model_score, paths['DEFAULT']['TRAINING_PATH'], config_params, **hyperparams)
    return model_score, model_name

# wrapper for hyperparameter optimization
def modelFitnessScore(paths, config_params, verbose=0, **hyperparams):
    score, _ = trainXValModel(paths, config_params, save_model=False, verbose=verbose, **hyperparams)
    return - score # since minimization

# wrapper for hyperparameter optimization
def singleModelFitnessScore(paths, config_params, model_id=1, verbose=0, **hyperparams):

    model_score, model_name = trainModel(paths=paths,
                                         config_params=config_params,
                                         model_id=model_id,
                                         save_model=False,
                                         verbose=verbose,
                                         **hyperparams)

    logTraining(model_name, model_score, paths['DEFAULT']['TRAINING_PATH'], config_params, **hyperparams)
    score = - model_score # since minimization
    return score

def evaluateModel(model_name, paths, config_params, verbose=0):

    params = config_params
    dataset_id = params['dataset']

    backbone = model_name.split('_')[0]


    training_dir = paths['DEFAULT']['TRAINING_PATH']

    # get model files
    model_file = []
    for file in os.listdir(training_dir):
        if file.startswith(model_name) and file.endswith('h5'):
            model_file = (os.path.join(training_dir, file))
    if verbose:
        print('evaluating {}'.format(model_name))

    # dataset
    # ================================================================================================
    dataset = getattr(datasets, dataset_id)(dataset_dir=paths['DEFAULT']['DATASETS_PATH'], verbose=verbose)

    preprocessing = getattr(utilities.dataset_utils,
                            params['test']['preprocessing'])(
        **configs.default.backbone_models[backbone]['input_parameters'])



    eval_data = dataset.makeBatch(subset= 'eval',
                                 preprocess_fn= preprocessing,
                                 batch_size=params['validation']['batch_size'])



    model = tf.keras.models.load_model(model_file)
    metric_callback = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile( # List of metrics to monitor
                  metrics=metric_callback)


    results = model.evaluate(eval_data)

    return results

def inferenceModel(model_name, paths,  eval_data):

    training_dir = paths['DEFAULT']['TRAINING_PATH']

    # get model files
    model_file = []
    for file in os.listdir(training_dir):
        if file.startswith(model_name) and file.endswith('h5'):
            model_file = (os.path.join(training_dir, file))


    model = tf.keras.models.load_model(model_file)

    output_class = model.predict(eval_data)

    return tf.argmax(output_class,axis=-1)