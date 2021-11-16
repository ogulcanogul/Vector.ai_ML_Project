import tensorflow as tf

# samplers for the datasets
# ==============================================================================

# base class for samplers
class Sampler(object):
    '''def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            '__init__() is to be implemented in Sampler sub classes')'''

    def __call__(self, dataset):
        raise NotImplementedError(
            '__call__() is to be implemented in Sampler sub classes')

# random samping
class Random(Sampler):
    def __init__(self, batch_size=32, shuffling_buffer_size=10000, random_seed=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffling_buffer_size = shuffling_buffer_size
        self.random_seed = random_seed

    def __call__(self, dataset):

        dataset = tf.data.Dataset.from_tensor_slices([ds.strip() for ds in dataset])

        dataset = dataset.shuffle(buffer_size=self.shuffling_buffer_size, seed=self.random_seed).repeat()

        dataset = dataset.batch(self.batch_size)

        return dataset


