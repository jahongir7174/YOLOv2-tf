import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from utils import config


class SpaceToDepth(layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        batch, height, width, depth = backend.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = backend.reshape(x, (batch, reduced_height, self.block_size, reduced_width, self.block_size, depth))
        z = backend.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        return backend.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
                 input_shape[1] // self.block_size,
                 input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return tf.TensorShape(shape)


def build_model(inputs, nb_class, trainable):
    x = layers.Conv2D(32, (3, 3), (1, 1), 'same', name='conv1', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), (1, 1), 'same', name='conv2', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), (1, 1), 'same', name='conv3', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(64, (1, 1), (1, 1), 'same', name='conv4', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(128, (3, 3), (1, 1), 'same', name='conv5', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), (1, 1), 'same', name='conv6', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn6')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(128, (1, 1), (1, 1), 'same', name='conv7', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn7')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(256, (3, 3), (1, 1), 'same', name='conv8', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn8')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(512, (3, 3), (1, 1), 'same', name='conv9', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn9')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(256, (1, 1), (1, 1), 'same', name='conv10', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn10')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(512, (3, 3), (1, 1), 'same', name='conv11', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn11')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(256, (1, 1), (1, 1), 'same', name='conv12', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn12')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(512, (3, 3), (1, 1), 'same', name='conv13', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn13')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    skip = x

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv14', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn14')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(512, (1, 1), (1, 1), 'same', name='conv15', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn15')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv16', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn16')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(512, (1, 1), (1, 1), 'same', name='conv17', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn17')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv18', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn18')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv19', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn19')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv20', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn20')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    skip = layers.Conv2D(64, (1, 1), (1, 1), 'same', name='conv21', kernel_regularizer=regularizers.l2(0.0005),
                         use_bias=False)(skip)
    skip = layers.BatchNormalization(name='bn21')(skip)
    skip = layers.LeakyReLU(alpha=0.1)(skip)

    skip = SpaceToDepth(block_size=2)(skip)

    x = layers.concatenate([skip, x])

    x = layers.Conv2D(1024, (3, 3), (1, 1), 'same', name='conv22', kernel_regularizer=regularizers.l2(0.0005),
                      use_bias=False)(x)
    x = layers.BatchNormalization(name='bn22')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(len(config.anchors) // 2 * (5 + nb_class), 1, 1, 'same', name='conv23', trainable=trainable)(x)
    x = layers.Reshape((config.image_size // config.scale,
                        config.image_size // config.scale,
                        len(config.anchors) // 2, (5 + nb_class)))(x)
    return x


def detection_loss(mask, boxes, one_hot, grid, y_pred):
    anchors = config.anchors
    size = config.image_size // config.scale
    anchors = anchors.reshape(len(anchors) // 2, 2)

    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(size), [size]), (1, size, size, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
    coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, len(anchors), 1])

    pred_xy = backend.sigmoid(y_pred[:, :, :, :, 0:2])
    pred_xy = (pred_xy + coords)
    pred_wh = backend.exp(y_pred[:, :, :, :, 2:4]) * anchors
    nb_mask = backend.sum(tf.cast(mask > 0.0, tf.float32))
    xy_loss = backend.sum(mask * backend.square(boxes[..., :2] - pred_xy)) / (nb_mask + 1e-6)
    wh_loss = backend.sum(mask * backend.square(backend.sqrt(boxes[..., 2:4]) - backend.sqrt(pred_wh))) / (
            nb_mask + 1e-6)
    coord_loss = xy_loss + wh_loss

    pred_box_class = y_pred[..., 5:]
    true_box_class = tf.argmax(one_hot, -1)
    class_loss = backend.sparse_categorical_crossentropy(true_box_class, pred_box_class, True)
    class_loss = backend.expand_dims(class_loss, -1) * mask
    class_loss = backend.sum(class_loss) / (nb_mask + 1e-6)

    pred_conf = backend.sigmoid(y_pred[..., 4:5])
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    w1 = boxes[..., 2]
    h1 = boxes[..., 3]
    x2 = pred_xy[..., 0]
    y2 = pred_xy[..., 1]
    w2 = pred_wh[..., 0]
    h2 = pred_wh[..., 1]

    x_min_1 = x1 - 0.5 * w1
    x_max_1 = x1 + 0.5 * w1
    y_min_1 = y1 - 0.5 * h1
    y_max_1 = y1 + 0.5 * h1
    x_min_2 = x2 - 0.5 * w2
    x_max_2 = x2 + 0.5 * w2
    y_min_2 = y2 - 0.5 * h2
    y_max_2 = y2 + 0.5 * h2
    intersection_x = backend.minimum(x_max_1, x_max_2) - backend.maximum(x_min_1, x_min_2)
    intersection_y = backend.minimum(y_max_1, y_max_2) - backend.maximum(y_min_1, y_min_2)
    intersection = intersection_x * intersection_y
    union = w1 * h1 + w2 * h2 - intersection
    iou = intersection / (union + 1e-6)
    iou = backend.expand_dims(iou, -1)

    pred_xy = backend.expand_dims(pred_xy, 4)
    pred_wh = backend.expand_dims(pred_wh, 4)
    pred_wh_half = pred_wh / 2.
    pred_min = pred_xy - pred_wh_half
    pred_max = pred_xy + pred_wh_half
    true_boxes_shape = backend.int_shape(grid)
    grid = backend.reshape(grid, [true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]])
    true_xy = grid[..., 0:2]
    true_wh = grid[..., 2:4]
    true_wh_half = true_wh * 0.5
    true_min = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    intersection_min = backend.maximum(pred_min, true_min)
    intersection_max = backend.minimum(pred_max, true_maxes)
    intersect_wh = backend.maximum(intersection_max - intersection_min, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    best_iou = backend.max(iou_scores, axis=4)
    best_iou = backend.expand_dims(best_iou)

    no_object_detection = backend.cast(best_iou < 0.6, backend.dtype(best_iou))
    no_obj_mask = no_object_detection * (1 - mask)
    nb_no_obj_mask = backend.sum(backend.cast(no_obj_mask > 0.0, 'float32'))

    no_object_loss = backend.sum(no_obj_mask * backend.square(-pred_conf)) / (nb_no_obj_mask + 1e-6)
    object_loss = backend.sum(mask * backend.square(iou - pred_conf)) / (nb_mask + 1e-6)
    conf_loss = no_object_loss + object_loss
    loss = conf_loss + class_loss + coord_loss
    return loss
