import os
from os.path import join

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import backend
from utils import util, config
from nets import nn


def get_box(y_pred, score_threshold, iou_threshold):
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(config.image_size // config.scale),
                                         [config.image_size // config.scale]),
                                 (1, config.image_size // config.scale,
                                  config.image_size // config.scale, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
    coords = tf.tile(tf.concat([coord_x, coord_y], -1), [1, 1, 1, len(config.anchors) // 2, 1])
    dims = backend.cast_to_floatx(backend.int_shape(y_pred)[1:3])
    dims = backend.reshape(dims, (1, 1, 1, 1, 2))
    anchors = config.anchors
    anchors = anchors.reshape(len(anchors) // 2, 2)
    pred_xy = backend.sigmoid(y_pred[:, :, :, :, 0:2])
    pred_xy = (pred_xy + coords)
    pred_xy = pred_xy / dims
    pred_wh = backend.exp(y_pred[:, :, :, :, 2:4])
    pred_wh = (pred_wh * anchors)
    pred_wh = pred_wh / dims
    box_conf = backend.sigmoid(y_pred[:, :, :, :, 4:5])
    box_class_prob = backend.softmax(y_pred[:, :, :, :, 5:])

    pred_xy = pred_xy[0, ...]
    pred_wh = pred_wh[0, ...]
    box_conf = box_conf[0, ...]
    box_class_prob = box_class_prob[0, ...]

    box_xy1 = pred_xy - 0.5 * pred_wh
    box_xy2 = pred_xy + 0.5 * pred_wh
    _boxes = backend.concatenate((box_xy1, box_xy2), -1)

    box_scores = box_conf * box_class_prob
    box_classes = backend.argmax(box_scores, -1)
    box_class_scores = backend.max(box_scores, -1)
    prediction_mask = box_class_scores >= score_threshold
    _boxes = tf.boolean_mask(_boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    _classes = tf.boolean_mask(box_classes, prediction_mask)

    _boxes = _boxes * config.image_size

    selected_idx = tf.image.non_max_suppression(_boxes, scores, config.max_boxes, iou_threshold)
    return backend.gather(_boxes, selected_idx), backend.gather(_classes, selected_idx)


if __name__ == '__main__':
    inputs = tf.keras.layers.Input((config.image_size, config.image_size, 3))
    outputs = nn.build_model(inputs, len(config.classes), False)
    model = tf.keras.models.Model(inputs, outputs)
    model.load_weights(join('weights', 'model15.h5'))
    path = '../Dataset/VOC2012/IMAGES/2007_000027.jpg'
    image = cv2.imread(path)
    image = image[:, :, ::-1]
    image = util.resize(image)
    input_image = image.astype('float32') / 127.5 - 1.0
    input_image = np.expand_dims(input_image, 0)
    boxes, classes = get_box(model.predict_on_batch(input_image), 0.45, 0.4)
    for i, box in enumerate(boxes):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        class_name = list(config.classes.keys())[list(config.classes.values()).index(classes[i].numpy())]
        cv2.putText(image, class_name, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    util.write_image('result.png', image)
