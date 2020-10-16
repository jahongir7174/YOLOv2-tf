import os
import sys
from os.path import exists
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(12345)
tf.random.set_seed(12345)

from nets import nn
from utils import config
from utils import data_loader
from utils import util

tf_path = join(config.data_dir, 'record')
tf_paths = [join(tf_path, name) for name in os.listdir(tf_path) if name.endswith('.tf')]

np.random.shuffle(tf_paths)

strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
global_batch = num_replicas * config.batch_size

steps = len(tf_paths) // global_batch

lr = nn.CosineLrSchedule(steps)

dataset = data_loader.TFRecordLoader(global_batch, config.epochs, len(config.classes)).load_data(tf_paths)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.935, decay=0.0005)
    inputs = tf.keras.layers.Input((config.image_size, config.image_size, 3))
    outputs = nn.build_model(inputs, len(config.classes), True)
    model = tf.keras.models.Model(inputs, outputs)
    model.summary()

weights_path = get_file(fname='yolov2.weights',
                        origin='https://pjreddie.com/media/files/yolov2.weights',
                        cache_subdir='models')
weight_reader = util.WeightReader(weights_path)
weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv' + str(i))
    conv_layer.trainable = True

    if i < nb_conv:
        norm_layer = model.get_layer('bn' + str(i))
        norm_layer.trainable = True

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])

layer = model.layers[-2]
layer.trainable = True

weights = layer.get_weights()
new_kernel = np.random.normal(size=weights[0].shape) / (
        config.image_size // config.scale * config.image_size // config.scale)
new_bias = np.random.normal(size=weights[1].shape) / (
        config.image_size // config.scale * config.image_size // config.scale)
layer.set_weights([new_kernel, new_bias])

print(f'[INFO] {len(tf_paths)} train data')

with strategy.scope():
    d_loss_object = nn.detection_loss


    def compute_loss(mask, box, one_hot, grid, y_pred):
        per_example_loss = d_loss_object(mask, box, one_hot, grid, y_pred)
        scale_loss = tf.reduce_sum(per_example_loss) * 1. / num_replicas

        return scale_loss

with strategy.scope():
    def train_step(image, mask, box, one_hot, grid):
        with tf.GradientTape() as tape:
            y_pred = model(image)

            loss = compute_loss(mask, box, one_hot, grid, y_pred)
        train_variable = model.trainable_variables
        gradient = tape.gradient(loss, train_variable)
        optimizer.apply_gradients(zip(gradient, train_variable))

        return loss

with strategy.scope():
    @tf.function
    def distribute_train_step(image, mask, box, one_hot, grid):
        loss = strategy.experimental_run_v2(train_step, args=(image,
                                                              mask,
                                                              box,
                                                              one_hot,
                                                              grid))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)


def main():
    print(f"--- Training with {steps} Steps ---")
    if not exists('weights'):
        os.makedirs('weights')
    for step, input_data in enumerate(dataset):
        step += 1
        image, mask, box, one_hot, grid = input_data
        loss = distribute_train_step(image, mask, box, one_hot, grid)
        print(f'{step} - {loss.numpy():.5f}')
        if step % steps == 0:
            model.save_weights(join("weights", f"model{step // steps}.h5"))
        if step // steps == config.epochs:
            sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
