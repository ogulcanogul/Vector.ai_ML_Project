import tensorflow as tf

from framework.layers import resMap, resBlock

def FashionResNet(input_shape,
                stages=3,
                blocks=2,
                block_repeats=1,
                dropout_probability=0.0,
                final_activation=True,
                num_classes=None,
                block_parameters=None):

    """ResNet Version 2 Model builder [b]
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
        # Arguments
            input_shape (tensor): shape of input image tensor
            num_stages (int): number of spatial decimation stages
            num_resnet_blocks (int): number of ReSNET blocks at each stage
        # Returns
            model (Model): Keras model instance
        """
    inputs = tf.keras.Input(shape=input_shape)

    # Start model definition.
    in_dim = 16

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    # first perform general feature extraction with convolution
    x = tf.keras.layers.Conv2D(filters=in_dim,
                               kernel_size=3,
                               strides=1,
                               padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)


    # Instantiate the stack of residual units
    out_dim = None
    pre_activation = block_parameters['pre_activation']
    for stage in range(stages):
        with tf.name_scope("stage{}".format(stage)):
            for res_block in range(blocks):
                with tf.name_scope("block{}".format(res_block)):
                    # general ReSNET bottleneck block configuration
                    block_parameters['pre_activation'] = pre_activation
                    spatial_decimation = 1
                    out_dim_reduction_rate = 2

                    # first stage has slightly different behavior
                    if stage == 0:
                        out_dim = in_dim * 4
                        out_dim_reduction_rate = 4
                        if res_block == 0:  # first layer and first stage
                            block_parameters['pre_activation'] = False
                    else:
                        out_dim = in_dim * 2

                        if res_block == 0:  # first layer but not first stage
                            spatial_decimation = 2  # downsample


                    # if first layer of a stage
                    if res_block == 0:
                        # linear projection residual shortcut connection
                        x = resMap(out_dim=out_dim,
                                   out_dim_reduction_rate=out_dim_reduction_rate,
                                   spatial_decimation=spatial_decimation,
                                   **block_parameters)(x)
                    else:
                        x = resBlock(out_dim=out_dim,
                                     out_dim_reduction_rate=2,
                                     repeats=block_repeats,
                                     **block_parameters)(x)

                    #dropout
                    x = tf.keras.layers.Dropout(dropout_probability)(x)

        in_dim = out_dim

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if final_activation:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    outputs = x
    if num_classes is not None:

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                                        activation='softmax',
                                        kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    return model
