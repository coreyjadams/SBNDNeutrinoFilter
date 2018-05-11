import sys
import time


import tensorflow as tf

from utils import residual_block, downsample_block

from resnetcore import resnetcore

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnet(resnetcore):
    '''Define a network model and run training

    U resnet implementation
    '''
    def __init__(self, params):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''

        super(resnet, self).__init__(params)


    def _build_network(self, input_placeholder):

        x = input_placeholder

        # The filters are concatenated at some point, and progress together

        verbose = True

        if verbose:
            print "Initial shape: " + str(x.get_shape())
        n_planes = self._params['NPLANES']

        x = tf.split(x, n_planes*[1], -1)
        if verbose:
            for p in range(len(x)):
                print "Plane {0} initial shape:".format(p) + str(x[p].get_shape())

        # Initial convolution to get to the correct number of filters:
        for p in range(len(x)):
            x[p] = tf.layers.conv2d(x[p], self._params['N_INITIAL_FILTERS'],
                                    kernel_size=[7, 7],
                                    strides=[2, 2],
                                    padding='same',
                                    use_bias=False,
                                    trainable=self._params['TRAINING'],
                                    name="Conv2DInitial_plane{0}".format(p),
                                    reuse=None)

            # ReLU:
            x[p] = tf.nn.relu(x[p])

        if verbose:
            print "After initial convolution: "

            for p in range(len(x)):
                print "Plane {0}".format(p) + str(x[p].get_shape())

        for p in xrange(len(x)):
          x[p] = residual_block(x[p], self._params['TRAINING'],
                                      batch_norm=True,
                                      name="initial_resblock1_plane{0}".format(p))
          x[p] = residual_block(x[p], self._params['TRAINING'],
                                      batch_norm=True,
                                      name="initial_resblock2_plane{0}".format(p))

        # Begin the process of residual blocks and downsampling:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH_PRE_MERGE']):

                x[p] = downsample_block(x[p], self._params['TRAINING'],
                                        batch_norm=True,
                                        name="downsample_plane{0}_{1}".format(p,i))

                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=True,
                                          name="resblock_plane{0}_{1}_{2}".format(p, i, j))
                if verbose:
                    print "Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                        p=p, i=i, s=x[p].get_shape())

        # print "Reached the deepest layer."

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=-1)

        if verbose:
            print "Shape after concatenation: " + str(x.get_shape())

        n_max_filters = self._params['NPLANES']*self._params['N_INITIAL_FILTERS']
        n_max_filters *= 2**self._params['NETWORK_DEPTH_PRE_MERGE']

        # At the bottom, do another residual block:
        for i in xrange(self._params['NETWORK_DEPTH_POST_MERGE']):

            x = downsample_block(x, self._params['TRAINING'],
                                 batch_norm=True,
                                 name="downsample_postmerge{0}".format(i))

            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                    batch_norm=True, name="resblock_postmerge_{0}_{1}".format(i, j))

            # Apply bottlenecking here to keep the number of filters in check:

            x = tf.layers.conv2d(x,
                         n_max_filters,
                         kernel_size=[1,1],
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=False,
                         trainable=self._params['TRAINING'],
                         name="Bottleneck_downsample_{0}".format(i))

        if verbose:
            print "Shape after final block: " + str(x.get_shape())



        # Apply a bottle neck to get the right shape:
        x = tf.layers.conv2d(x,
                         self._params['NUM_LABELS'],
                         kernel_size=[1,1],
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=False,
                         trainable=self._params['TRAINING'],
                         name="BottleneckConv2D")

        if verbose:
            print "Shape after bottleneck: " + str(x.get_shape())

        # Apply global average pooling to get the right final shape:
        shape = (x.shape[1], x.shape[2])
        x = tf.nn.pool(x,
                   window_shape=shape,
                   pooling_type="AVG",
                   padding="VALID",
                   dilation_rate=None,
                   strides=None,
                   name="GlobalAveragePool",
                   data_format=None)

        if verbose:
            print "Shape after pooling: " + str(x.get_shape())


        # Reshape to the right shape for logits:
        x = tf.reshape(x, [tf.shape(x)[0], self._params['NUM_LABELS']],
                 name="global_pooling_reshape")

        if verbose:
            print "Final shape: " + str(x.get_shape())




        return x
