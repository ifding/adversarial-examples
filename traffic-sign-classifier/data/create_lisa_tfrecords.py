
"""
Convert the LISA Traffic Sign dataset into Tensorflow tfrecords.

1. Download LISA Dataset here : http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
2. Specify dataset root directory and Output directory

Example usage:
    ./create_lisa_tf_record --data_dir=/home/user/lisa \
        --output_dir=/home/user/lisa/output

"""

import csv
import cv2

import hashlib
import logging
import os
import random
import re
from random import shuffle
import glob
import numpy as np
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw LISA dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/lisa_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

data_dir = '/scratch3/feid/feid-data/LISA-data/'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def create_tf_record(output_filename, examples, labels, data_type):
    """Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
            print ('On image %d of %d'%( idx, len(examples)))
        example = example[0]
        label = example[1]
        label = labels.index(label)
        image_path = os.path.join(data_dir, example[0])

        if not os.path.exists(image_path):
            logging.warning('Could not find %s, ignoring example.', image_path)
            continue
    
        # Load the image
        img = load_image(image_path)

        # Create a feature
    
        if data_type == 'train':
            feature = {'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        else:
            feature = {'val/label': _int64_feature(label),
                       'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}        
        # Create an example protocol buffer
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))        
    
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
  
    data_dir = FLAGS.data_dir

    logging.info('Reading from LISA dataset.')
  
    annotations_dir = os.path.join(data_dir, 'allAnnotations.csv')
    labels = []
    file = open('categories.txt', 'r') 
    for line in file: 
        line = line.strip('\n')
        labels.append(line) 

    with open(annotations_dir) as csvFile :
        datareader = csv.reader(csvFile, delimiter = ';')
        next(datareader) # for skipping first row
        parse_data = []
        for row in datareader:
            parse_data.append([row])
  

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split. This happens randomly

    random.seed(49)
    random.shuffle(parse_data)
    num_examples = len(parse_data)

    num_train = int(0.9 * num_examples)
    train_examples = parse_data[:num_train]
    val_examples = parse_data[num_train:]
    logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'lisa_train.tfrecords')
    val_output_path = os.path.join(FLAGS.output_dir, 'lisa_val..tfrecords')
    create_tf_record(train_output_path, train_examples, labels, "train")
    create_tf_record(val_output_path, val_examples, labels, "val")

if __name__ == '__main__':
      tf.app.run()
