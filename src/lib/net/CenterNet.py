# Construct the tensorflow model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
const_init = tf.compat.v1.constant_initializer


def batch_init(weights, layer):
    init = {
        'gamma': const_init(weights[layer + '.weight'], ),
        'beta': const_init(weights[layer + '.bias']),
        'moving_mean': const_init(weights[layer + '.running_mean']),
        'moving_variance': const_init(weights[layer + '.running_var'])
    }
    return init


def explict_padding(input, p):
    return tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]])


def bn(input, weights, name):
    return slim.batch_norm(input,
                           center=True,
                           scale=True,
                           epsilon=1e-05,
                           is_training=False,
                           param_initializers=batch_init(weights, name))
def conv_transpose(input, weights, o, k, s, weight_name = None):
    weight_init = None
    if weight_name:
        weight_init = const_init(weights[weight_name])
    return slim.conv2d_transpose(input,
                                 num_outputs=o,
                                 kernel_size=k,
                                 stride=s,
                                 padding='SAME',
                                 weights_initializer=weight_init,
                                 biases_regularizer=None,
                                 activation_fn=None)


def conv(input, weights, o, k, s, p=None, weight_name=None, bias_name=None):
    weight_init = None
    bias_init = None
    use_bias = False
    if p:
        input = explict_padding(input, p)
    if weight_name:
        weight_init = const_init(weights[weight_name])
    if bias_name:
        bias_init = const_init(weights[bias_name])
        use_bias = True
    return slim.conv2d(input,
                       o, [k, k],
                       padding='VALID',
                       stride=s,
                       weights_initializer=weight_init,
                       biases_initializer=bias_init)


def CenterNet(inputs, weights, spatial_squeeze=True, scope='CenterNet'):
    with tf.compat.v1.variable_scope(scope, 'CenterNet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            padding='VALID',
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=None,
                            outputs_collections=[end_points_collection]):
            outputs = {}
            outputs['inputs'] = inputs

            net = conv(inputs, weights, 64, 7, 2, 3, 'conv1.weight')
            net = bn(net, weights, 'bn1')
            net = tf.nn.relu(net)
            net = explict_padding(net, 1)
            net = slim.max_pool2d(net, [3, 3], stride=2)
            
            # Layer 1 - block 1
            residual = net
            net = conv(net, weights, 64, 3, 1, 1, 'layer1.0.conv1.weight')
            net = bn(net, weights, 'layer1.0.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 64, 3, 1, 1, 'layer1.0.conv2.weight')
            net = bn(net, weights, 'layer1.0.bn2')
            net += residual
            net = tf.nn.relu(net)

            # Layer 1 - block 2
            residual = net
            net = conv(net, weights, 64, 3, 1, 1, 'layer1.1.conv1.weight')
            net = bn(net, weights, 'layer1.1.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 64, 3, 1, 1, 'layer1.1.conv2.weight')
            net = bn(net, weights, 'layer1.1.bn2')
            net += residual
            net = tf.nn.relu(net)
 
            # Layer 2 - block 1
            residual = net
            net = conv(net, weights, 128, 3, 2, 1, 'layer2.0.conv1.weight')
            net = bn(net, weights, 'layer2.0.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 128, 3, 1, 1, 'layer2.0.conv2.weight')
            net = bn(net, weights, 'layer2.0.bn2')

            residual = conv(residual, weights, 128, 1, 2, None,
                            'layer2.0.downsample.0.weight')
            residual = bn(residual, weights, 'layer2.0.downsample.1')
            net += residual
            net = tf.nn.relu(net)

            # Layer 2 - block 2
            residual = net

            net = conv(net, weights, 128, 3, 1, 1, 'layer2.1.conv1.weight')
            net = bn(net, weights, 'layer2.1.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 128, 3, 1, 1, 'layer2.1.conv2.weight')
            net = bn(net, weights, 'layer2.1.bn2')
            net += residual
            net = tf.nn.relu(net)

            # Layer 3 - block 1
            residual = net

            net = conv(net, weights, 256, 3, 2, 1, 'layer3.0.conv1.weight')
            net = bn(net, weights, 'layer3.0.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 256, 3, 1, 1, 'layer3.0.conv2.weight')
            net = bn(net, weights, 'layer3.0.bn2')

            residual = conv(residual, weights, 256, 1, 2, None,
                            'layer3.0.downsample.0.weight')
            residual = bn(residual, weights, 'layer3.0.downsample.1')

            net += residual
            net = tf.nn.relu(net)

            # Layer 3 - block 2
            residual = net

            net = conv(net, weights, 256, 3, 1, 1, 'layer3.1.conv1.weight')
            net = bn(net, weights, 'layer3.1.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 256, 3, 1, 1, 'layer3.1.conv2.weight')
            net = bn(net, weights, 'layer3.1.bn2')
            net += residual
            net = tf.nn.relu(net)
 
            # Layer 4 - block 1
            residual = net

            net = conv(net, weights, 512, 3, 2, 1, 'layer4.0.conv1.weight')
            net = bn(net, weights, 'layer4.0.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 512, 3, 1, 1, 'layer4.0.conv2.weight')
            net = bn(net, weights, 'layer4.0.bn2')

            residual = conv(residual, weights, 512, 1, 2, None,
                            'layer4.0.downsample.0.weight')
            residual = bn(residual, weights, 'layer4.0.downsample.1')

            net += residual
            net = tf.nn.relu(net)

            # Layer 4 - block 2
            residual = net

            net = conv(net, weights, 512, 3, 1, 1, 'layer4.1.conv1.weight')
            net = bn(net, weights, 'layer4.1.bn1')
            net = tf.nn.relu(net)

            net = conv(net, weights, 512, 3, 1, 1, 'layer4.1.conv2.weight')
            net = bn(net, weights, 'layer4.1.bn2')

            net += residual
            net = tf.nn.relu(net)

            # Deconv Layer
            # Layer 1
            net = conv_transpose(net, weights, 256, 4, 2, weight_name = 'deconv_layers.0.weight')
            net = bn(net, weights, 'deconv_layers.1')
            net = tf.nn.relu(net)

            # Layer 2
            net = conv_transpose(net, weights, 256, 4, 2, weight_name = 'deconv_layers.3.weight')
            net = bn(net, weights, 'deconv_layers.4')
            net = tf.nn.relu(net)

            # Layer 3
            net = conv_transpose(net, weights, 256, 4, 2, weight_name = 'deconv_layers.6.weight')
            net = bn(net, weights, 'deconv_layers.7')
            net = tf.nn.relu(net)
            outputs['Layer6'] = net
            
            # Heatmap
            # Remark: classes here can change, 10 is for bdd dataset
            classes = 10

            hm = conv(net, weights, 64, 3, 1, 1, 'hm.0.weight', 'hm.0.bias')
            hm = slim.nn.relu(hm)
            hm = conv(hm, weights, classes, 1, 1, None, 'hm.2.weight',
                      'hm.2.bias')

            # Add sigmoid and nms (original in process function of ctdet)

            hm = slim.nn.sigmoid(hm)
            hm_max = explict_padding(hm, 1)
            hm_max = slim.max_pool2d(hm_max, [3, 3], stride=1, padding='VALID')
            keep = tf.equal(hm, hm_max)
            keep = tf.cast(keep, tf.float32)
            hm_keep = tf.multiply(hm, keep)
            outputs['hm'] = hm_keep
            # WH
            wh = conv(net, weights, 64, 3, 1, 1, 'wh.0.weight', 'wh.0.bias')
            wh = slim.nn.relu(wh)
            wh = conv(wh, weights, 2, 1, 1, None, 'wh.2.weight', 'wh.2.bias')
            # reg
            reg = conv(net, weights, 64, 3, 1, 1, 'reg.0.weight', 'reg.0.bias')
            reg = slim.nn.relu(reg)
            reg = conv(reg, weights, 2, 1, 1, None, 'reg.2.weight',
                       'reg.2.bias')
            outputs['reg'] = reg
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)

            if spatial_squeeze:
                hm_keep = tf.squeeze(hm_keep, [0], name='hm')
                wh = tf.squeeze(wh, [0], name='wh')
                reg = tf.squeeze(reg, [0], name='reg')
                end_points['hm'] = hm_keep
                end_points['wh'] = wh
                end_points['reg'] = reg
        return net, end_points, outputs
