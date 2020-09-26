import tensorflow as tf
from utils import config


class TFRecordLoader:
    def __init__(self, batch_size, nb_epoch, nb_class=20):
        super(TFRecordLoader, self).__init__()
        self.nb_epoch = nb_epoch
        self.nb_classes = nb_class
        self.batch_size = batch_size
        self.feature_description = {'path': tf.io.FixedLenFeature([], tf.string),
                                    'box': tf.io.FixedLenFeature([], tf.string),
                                    'mask': tf.io.FixedLenFeature([], tf.string),
                                    'grid': tf.io.FixedLenFeature([], tf.string)}

    def parse_data(self, tf_record):
        features = tf.io.parse_single_example(tf_record, self.feature_description)

        image = tf.io.read_file(features['path'])
        image = tf.io.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)
        image = image / 255.0

        mean = tf.constant([0.485, 0.456, 0.406])
        mean = tf.expand_dims(mean, axis=0)
        mean = tf.expand_dims(mean, axis=0)
        image -= mean

        std = tf.constant([0.229, 0.224, 0.225])
        std = tf.expand_dims(std, axis=0)
        std = tf.expand_dims(std, axis=0)
        image /= std

        box = tf.io.decode_raw(features['box'], tf.float32)
        box = tf.reshape(box, (config.image_size // config.scale,
                               config.image_size // config.scale,
                               len(config.anchors) // 2, 5))

        mask = tf.io.decode_raw(features['mask'], tf.float32)
        mask = tf.reshape(mask, (
            config.image_size // config.scale, config.image_size // config.scale, len(config.anchors) // 2, 1))

        grid = tf.io.decode_raw(features['grid'], tf.float32)
        grid = tf.reshape(grid, (config.max_boxes, 5))

        one_hot = tf.one_hot(tf.cast(tf.expand_dims(box[..., 4], 0), 'int32'), self.nb_classes + 1)[:, :, :, :, 1:]
        one_hot = tf.squeeze(one_hot, 0)
        one_hot = tf.cast(one_hot, dtype='float32')

        return image, mask, box, one_hot, grid

    def load_data(self, file_names):
        reader = tf.data.TFRecordDataset(file_names)
        reader = reader.shuffle(len(file_names))
        reader = reader.map(self.parse_data, tf.data.experimental.AUTOTUNE)
        reader = reader.repeat(self.nb_epoch + 1)
        reader = reader.batch(self.batch_size)
        reader = reader.prefetch(tf.data.experimental.AUTOTUNE)
        return reader
