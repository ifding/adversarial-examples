import tensorflow as tf
import numpy as np


class PGD_attack:
    """ Creates adversarial samples using a projected gradient descent attack
    """
    def __init__(self, model, 
                 batch_shape, 
                 max_epsilon, 
                 max_iter, 
                 targeted, 
                 img_bounds=[-1, 1],
                 use_noise=True,
                 initial_lr=0.5, 
                 lr_decay=0.98,
                 n_classes=1001,
                 rng = np.random.RandomState()):
        """ 
             model: Callable (function) that accepts an input tensor 
                    and return the model logits (unormalized log probs)
             batch_shape: Input shapes (tuple). 
                    Usually: [batch_size, height, width, channels]
             max_epsilon: Maximum L_inf norm for the adversarial example
             max_iter: Maximum number of gradient descent iterations
             targeted: Boolean: true for targeted attacks, false for non-targeted attacks
             img_bounds: Tuple [min, max]: bounds of the image. Example: [0, 255] for
                    a non-normalized image, [-1, 1] for inception models.
             initial_lr: Initial Learning rate for the optimization
             lr_decay: Learning rate decay (multiplied in the lr in each iteration)
             rng: Random number generator 
             
        """
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int32, shape=(batch_shape[0]))
        
        # Loss function: the mean of the logits of the correct class
        y_onehot = tf.one_hot(self.y_input, n_classes)
        logits = model(self.x_input)
        logits_correct_class = tf.reduce_sum(logits * y_onehot, axis=1)

        self.loss = tf.reduce_mean(logits_correct_class)
        self.grad = tf.gradients(self.loss, self.x_input)

        # Keep track of the parameters:
        self.targeted = targeted
        self.max_iter = max_iter
        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        self.use_noise = use_noise
        self.rng = rng
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        
    def generate(self, sess, images, labels_or_targets, verbose=False):
        """ Generates adversarial images/
            sess: the tensorflow session
            images: a 4D tensor containing the original images
            labels_or_targets: for non-targeted attacks, the actual or predicted labels
                               for targeted attacks, the desired target classes for each image.
            
            returns: adv_images: a 4D tensor containing adversarial images
        """
        if self.use_noise:
            # Random starting step, from https://arxiv.org/abs/1705.07204
            alpha = self.max_epsilon * 0.5
            delta_init = alpha * np.sign(self.rng.normal(size=np.shape(images))).astype(np.float32)
        else:
            # Or start from the original image (i.e. no perturbation)
            delta_init = np.zeros(np.shape(images), dtype=np.float32)
            

        lr = self.initial_lr
        delta = delta_init
        
        if self.targeted:
            multiplier = 1. # For targeted attack: maximize logits of desired class
        else:
            multiplier = -1. # For non-targeted attack: minimize logits of correct class
            
        # Calculate the bounds for the perturbation
        lower_bounds = np.maximum(self.img_bounds[0] - images, -self.max_epsilon)
        upper_bounds = np.minimum(self.img_bounds[1] - images, self.max_epsilon)
        
       
        for i in range(self.max_iter):
            l, gradients  = sess.run([self.loss, self.grad], 
                                 feed_dict={self.x_input:images + delta,
                                            self.y_input:labels_or_targets})
            
            delta = delta + multiplier * lr * gradients[0]
            
            # Project delta to the region that satisfy the constraints
            delta = np.clip(delta, lower_bounds, upper_bounds)
            
            lr = lr * self.lr_decay
            
            if verbose:
                print('Iter %d, loss: %.2f' % (i, l))
        return images + delta
