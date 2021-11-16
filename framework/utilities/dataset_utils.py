import tensorflow as tf

def resizeImageKeepAspectRatio(image_tensor, target_size):

    lo_dim = tf.cast(tf.reduce_max(target_size), tf.float32)

    # Take width/height
    initial_width = tf.shape(image_tensor)[0]
    initial_height = tf.shape(image_tensor)[1]

    # Take the greater value, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.cast(min_, tf.float32) / lo_dim #tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.cast(tf.cast(initial_width, tf.float32) / ratio, tf.int32)
    new_height = tf.cast(tf.cast(initial_height, tf.float32) / ratio, tf.int32)

    return tf.image.resize(image_tensor, [new_width, new_height], method=tf.image.ResizeMethod.BILINEAR)

class ImagePreprocessing(object):

    def __init__(self, **kwargs):
        pass

    @tf.function
    def __call__(self, image, training=tf.constant(False, dtype=tf.bool), **kwargs):

        image = tf.cond(training,
                        true_fn=lambda: self.trainPreprocess(image, **kwargs),
                        false_fn=lambda: self.testPreprocess(image, **kwargs))

        return image

    def trainPreprocess(self, image, **kwargs):
        raise NotImplementedError(
            'trainPreprocess() is to be implemented in ImagePreprocessing sub classes')

    def testPreprocess(self, image, **kwargs):
        raise NotImplementedError(
            'testPreprocess() is to be implemented in ImagePreprocessing sub classes')

class FashionMNISTPreprocessing(ImagePreprocessing):
    def __init__(self, out_image_size=(28, 28), **kwargs):
        super(FashionMNISTPreprocessing, self).__init__(**kwargs)
        self._out_image_size = out_image_size
        self.out_image_size = list(self._out_image_size) + [1]

    def trainPreprocess(self, image, **kwargs):
        """Preprocess a single image in [height, width, depth] layout."""
        # Pad 4 pixels on each dimension of feature map, done in mini-batch

        image_tensor = tf.cast(image,tf.float32)
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, self._initial_resize[0], self._initial_resize[1])
        #image_tensor = tf.image.random_crop(image_tensor, list(self._out_image_size))
        #image_tensor = tf.image.random_flip_left_right(image_tensor)

        if len(image_tensor.shape) == 3:
            image_tensor = tf.reduce_mean(image_tensor, axis=-1, keepdims=True)

        return image_tensor
    def testPreprocess(self, image, **kwargs):
        """pass image as is"""

        image_tensor = tf.cast(image,tf.float32)
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, 40, 40)
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._out_image_size)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.reduce_mean(image_tensor, axis=-1, keepdims=True)

        return image_tensor




