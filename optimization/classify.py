#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

from models.helper import get_data_spec, get_model
from models.googlenet import GoogleNet
import dataset


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('data/imagenet-classes.txt', 'rb') as f:
        classes = f.read().splitlines()
        class_labels = np.array(classes)
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def classify(model_data_path, model_name, image_paths, use_crop):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    spec = get_data_spec(model_name)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = GoogleNet({'data': input_node})

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.ImageProducer(image_paths=image_paths, 
                                          data_spec=spec,
                                          need_rescale=use_crop,
                                          batch_size=1)

    with tf.Session() as sesh:
        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        indices, input_images = image_producer.get_batch(sesh)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
        display_results([image_paths[i] for i in indices], probs)

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the GoogleNet model')
    parser.add_argument('--model', type=str, required=True,choices=['GoogleNet'],
                        help='Models to be evaluated.')
    parser.add_argument('image_paths', nargs='+', help='One or more images to classify')
    parser.add_argument('--not_crop', dest='use_crop', action='store_false',
                        help='Not use crop in image producer.')
    args = parser.parse_args()

    # Classify the image
    classify(args.model_path, args.model, args.image_paths, args.use_crop)


if __name__ == '__main__':
    main()
