from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import tensorflow as tf


class lbfgs_attack:
    """ Creates adversarial samples using box contrained L-BFGS
    """

    def __init__(self,
                 model,
                 batch_shape,
                 max_epsilon,
                 max_iter,
                 targeted,
                 img_bounds=(-1, 1),
                 use_noise=True,
                 max_ls=5,
                 n_classes=1001,
                 rng=np.random.RandomState()):
        """ 
             model: Callable (function) that accepts an input tensor 
                    and return the model logits (unormalized log probs)
             batch_shape: Input shapes (tuple). 
                    Usually: [batch_size, height, width, channels]
             max_epsilon: Maximum L_inf norm for the adversarial example
             max_iter: Maximum number of iterations (gradient computations)
             targeted: Boolean: true for targeted attacks, false for non-targeted attacks
             img_bounds: Tuple [min, max]: bounds of the image. Example: [0, 255] for
                    a non-normalized image, [-1, 1] for inception models.
             max_ls: Maximum number of line searches
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

        self.targeted = targeted
        self.max_iter = max_iter
        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        self.use_noise = use_noise
        self.rng = rng
        self.max_ls = max_ls

    def generate(self, sess, images, labels_or_targets, verbose=False):
        """ Generates adversarial images/
            sess: the tensorflow session
            images: a 4D tensor containing the original images
            labels_or_targets: for non-targeted attacks, the actual or predicted labels
                               for targeted attacks, the desired target classes for each image.

            returns: adv_images: a 4D tensor containing adversarial images
        """
        lower_bounds = np.maximum(-1 - images, -self.max_epsilon).reshape(-1)
        upper_bounds = np.minimum(1 - images, self.max_epsilon).reshape(-1)
        bounds = list(zip(lower_bounds, upper_bounds))

        def func(delta):
            attack_img = images + delta.reshape(images.shape).astype(np.float32)
            loss, gradients = sess.run([self.loss] + self.grad,
                                       feed_dict={self.x_input: attack_img,
                                                  self.y_input: labels_or_targets})
            if self.targeted:
                # Multiply by -1 since we want to maximize it.
                return -1 * loss, -1 * gradients.reshape(-1).astype(np.float)
            else:
                return loss, gradients.reshape(-1).astype(np.float)

        if self.use_noise:
            alpha = self.max_epsilon * 0.5
            x0 = alpha * np.sign(np.random.random(np.prod(images.shape)))
        else:
            x0 = np.zeros(np.prod(images.shape)),

        if verbose:
            iprint = 1
        else:
            iprint = -1

        delta_best, f, d = fmin_l_bfgs_b(func=func,
                                         x0=x0,
                                         bounds=bounds,
                                         maxfun=self.max_iter,
                                         maxls=self.max_ls,
                                         iprint=iprint)
        return images + delta_best.reshape(images.shape).astype(np.float32)
