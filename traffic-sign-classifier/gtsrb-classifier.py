import cv2
import os
import csv
import logging
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
import random
from sklearn.utils import shuffle
from tqdm import tqdm
from numpy import newaxis

#import matplotlib.pyplot as plt


train_path = '/scratch3/feid/feid-data/lisa_train.tfrecords'  # address to save the hdf5 file
val_path = '/scratch3/feid/feid-data/lisa_val.tfrecords'

data_dir = '/scratch3/feid/feid-data/LISA-data/'


EPOCHS = 10
BATCH_SIZE = 128
image_depth = 3
n_classes = 17


def load_image(addr):
    # read an image and resize to (32, 32)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img.reshape([1, 32, 32, 3])

def create_dataset(examples,labels):
    """Creates dataset from examples.
    """
    imgs, lbl = [], []
    for idx, example in enumerate(examples):
        if idx % 1000 == 0:
            logging.info('On image %d of %d', idx, len(examples))
            print ('On image %d of %d'%( idx, len(examples)))
        example = example[0]
        label = example[1]
        label = labels.index(label)
        image_path = os.path.join(data_dir, example[0])

        if not os.path.exists(image_path):
            #logging.warning('Could not find %s, ignoring example.', image_path)
            continue
    
        # Load the image
        img = load_image(image_path)
        
        if idx == 0:
            imgs = img
        else:
            imgs = np.vstack((imgs, img))
            
        lbl.append(label)
    return imgs, np.array(lbl)

# Iterates through grayscale for each image in the data
def preprocess(data):
    gray_images = []
    for image in data:
        gray = grayscale(image)
        gray_images.append(gray)
        
    return np.array(gray_images)

# Grayscales an image
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize(value):
    return value / 255 * 2 - 1

def preprocess_image(image):
    img = grayscale(image)
    img = normalize(img)
    return img

def preprocess_batch(images):
    imgs = np.zeros(shape=images.shape)
    processed_image_depth = preprocess_image(images[0]).shape[2]
    imgs = imgs[:,:,:,0:processed_image_depth]
    for i in tqdm(range(images.shape[0])):
        imgs[i] = preprocess_image(images[i])        
    return imgs


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
        
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


def main():
    
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
            if row[1] in labels:
                parse_data.append([row])
            
    random.seed(49)
    random.shuffle(parse_data)
    num_examples = len(parse_data)

    num_train = int(0.8 * num_examples)
    train_examples = parse_data[:num_train]
    val_examples = parse_data[num_train:]
    logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
    
    X_train, y_train = create_dataset(train_examples,labels)
    X_valid, y_valid = create_dataset(val_examples,labels)
    
    n_train = X_train.shape[0]
    n_valid = X_valid.shape[0]
    image_shape = X_train.shape[1:]

    n_classes = np.unique(y_train).size

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    
    # Iterate through grayscale
    X_train = preprocess(X_train)
    X_train = X_train[..., newaxis]

    # Normalize
    X_train = normalize(X_train) 
    
    # Iterate through grayscale
    X_valid = preprocess(X_valid)
    X_valid = X_valid[..., newaxis]

    # Normalize
    X_valid = normalize(X_valid) 
    
    # Double-check that the image is changed to depth of 1
    print("Processed training data shape =", X_train.shape)
    print("Processed testing data shape =", X_valid.shape)
    
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    
    # Set placeholder variables for x, y, and the keep_prob for dropout
    # Also, one-hot encode y
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    one_hot_y = tf.one_hot(y, 17)
    
    # Setting learning rate, loss functions, and optimizer
    rate = 0.005

    logits = neural_network(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    
    # The below is used in the validation part of the neural network
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    #X_train_processed = preprocess_batch(X_train)
    #X_valid_processed = preprocess_batch(X_valid)
    
    #x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
    #y = tf.placeholder(tf.int32, (None))
    #one_hot_y = tf.one_hot(y, n_classes)

    #rate = 0.003

    #logits = LeNet(x)
    #logits = model(x)

    #varss = tf.trainable_variables() 
    #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in varss
    #                if '_b' not in v.name ]) * 0.0001

    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    #loss_operation = tf.reduce_mean(cross_entropy) + lossL2
    #optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    #training_operation = optimizer.minimize(loss_operation)


    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    #accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #saver = tf.train.Saver()
      
    cost_arr = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        num_valids = len(X_valid) 
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            #X_trai, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                to, cost = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})      

            total_accuracy = 0
            for offset in range(0, num_valids, BATCH_SIZE):
                batch_v_x, batch_v_y = X_valid[offset:offset+BATCH_SIZE], y_valid[offset:offset+BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_v_x, y: batch_v_y})
                total_accuracy += (accuracy * len(batch_v_x))
            validation_accuracy = total_accuracy / num_valids
            
            print("EPOCH; {}; Valid.Acc.; {:.3f}; Loss; {:.5f}".format(i+1, validation_accuracy, cost))
            cost_arr.append(cost)
      
        
        saver.save(sess, './lenet')
        print("Model saved")


if __name__ == '__main__':
    main()