import copy
import multiprocessing
import os
from multiprocessing import Process
from multiprocessing import cpu_count
from os.path import exists
from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
import tqdm
from six import raise_from

from utils import config
from utils import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def name_to_label(name):
    return config.classes[name]


def load_image(f_name):
    path = join(config.data_dir, config.image_dir, f_name + '.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def parse_annotation(element):
    truncated = find_node(element, 'truncated', parse=int)
    difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.classes:
        raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(config.classes.keys())))

    label = config.classes[class_name]

    box = find_node(element, 'bndbox')
    x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=int)
    y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=int)
    x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=int)
    y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=int)

    return truncated, difficult, [x_min, y_min, x_max, y_max, label]


def parse_annotations(xml_root):
    annotations = []
    for i, element in enumerate(xml_root.iter('object')):
        truncated, difficult, box = parse_annotation(element)

        annotations.append(box)

    return np.array(annotations)


def load_label(f_name):
    try:
        tree = parse_fn(join(config.data_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)
    except ValueError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)


def byte_feature(value):
    import tensorflow as tf
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def build_example(f_name):
    import tensorflow as tf
    image = load_image(f_name)
    label = load_label(f_name)
    image, label = util.resize(image, label)
    boxes = np.zeros((1, config.max_boxes, 5))
    boxes[0, :label.shape[0], :5] = label
    boxes = boxes[0]
    anchors = copy.deepcopy(config.anchors)
    anchors = anchors.reshape(len(anchors) // 2, 2)

    nb_anchors = len(anchors)

    grid = np.zeros(boxes.shape)
    mask = np.zeros((config.image_size // config.scale, config.image_size // config.scale, nb_anchors, 1))
    true_boxes = np.zeros((config.image_size // config.scale, config.image_size // config.scale, nb_anchors, 5))

    for index, box in enumerate(boxes):
        w = (box[2] - box[0]) / config.scale
        h = (box[3] - box[1]) / config.scale
        x = ((box[0] + box[2]) / 2) / config.scale
        y = ((box[1] + box[3]) / 2) / config.scale
        grid[:box.shape[0], ...] = np.array([x, y, w, h, box[4]])
        if w * h > 0:
            best_iou = 0
            best_anchor = 0
            for i in range(nb_anchors):
                intersect = np.minimum(w, anchors[i, 0]) * np.minimum(h, anchors[i, 1])
                union = (anchors[i, 0] * anchors[i, 1]) + (w * h) - intersect
                iou = intersect / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            if best_iou > 0:
                x_coord = np.floor(x).astype('int')
                y_coord = np.floor(y).astype('int')
                mask[y_coord, x_coord, best_anchor] = 1
                true_boxes[y_coord, x_coord, best_anchor] = np.array([x, y, w, h, box[4]])

    path = join(config.data_dir, 'record', f_name + '.jpg')

    box = true_boxes.astype('float32')
    mask = mask.astype('float32')
    grid = grid.astype('float32')

    box = box.tobytes()
    mask = mask.tobytes()
    grid = grid.tobytes()

    features = tf.train.Features(feature={'path': byte_feature(path.encode('utf-8')),
                                          'box': byte_feature(box),
                                          'mask': byte_feature(mask),
                                          'grid': byte_feature(grid)})

    return tf.train.Example(features=features), image


def write_tf_record(_queue, _sentinel):
    import tensorflow as tf
    while True:
        f_name = _queue.get()

        if f_name == _sentinel:
            break
        tf_example, image = build_example(f_name)
        if not exists(join(config.data_dir, 'record')):
            os.makedirs(join(config.data_dir, 'record'))

        util.write_image(join(config.data_dir, 'record', f_name + '.jpg'), image)
        with tf.io.TFRecordWriter(join(config.data_dir, 'record', f_name + ".tf")) as writer:
            writer.write(tf_example.SerializeToString())


def main():
    f_names = []
    with open(join(config.data_dir, 'train.txt')) as reader:
        for line in reader.readlines():
            f_names.append(line.rstrip().split(' ')[0])
    sentinel = ("", [])
    queue = multiprocessing.Manager().Queue()
    for f_name in tqdm.tqdm(f_names):
        queue.put(f_name)
    for _ in range(cpu_count()):
        queue.put(sentinel)
    print('[INFO] generating TF record')
    process_pool = []
    for i in range(cpu_count()):
        process = Process(target=write_tf_record, args=(queue, sentinel))
        process_pool.append(process)
        process.start()
    for process in process_pool:
        process.join()


if __name__ == '__main__':
    main()
