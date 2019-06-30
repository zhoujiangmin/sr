import tensorflow as tf
import tensorflow.contrib.slim as slim


def deconv_and_concat(input, convx):
    b, out_h, out_w, out_c = convx.shape.as_list()
    _, _, _, in_c = input.shape.as_list()
    conv = slim.conv2d_transpose(input, out_c, [3, 3], stride=2)
    concat = tf.concat([convx, conv], axis=3)
    return concat

def UNet(input, scale, name='U-Net'):
    conv1 = slim.conv2d(input, scale*4, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv1_1'%name)
    conv1 = slim.conv2d(conv1, scale*4, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv1_2' % name)
    pool1 = slim.avg_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, scale*8, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv2_1'%name)
    conv2 = slim.conv2d(conv2, scale*8, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv2_2' % name)
    pool2 = slim.avg_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, scale*16, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv3_1'%name)
    conv3 = slim.conv2d(conv3, scale*16, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv3_2' % name)
    pool3 = slim.avg_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, scale*32, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv4_1'%name)
    conv4 = slim.conv2d(conv4, scale*32, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv4_2' % name)
    pool4 = slim.avg_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, scale*64, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv5_1'%name)
    conv5 = slim.conv2d(conv5, scale*64, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv5_2'%name)
    concat4_5 = deconv_and_concat(conv5, conv4)

    conv6 = slim.conv2d(concat4_5, scale*32, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv6_1'%name)
    conv6 = slim.conv2d(conv6, scale*32, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv6_2' % name)
    concat3_6 = deconv_and_concat(conv6, conv3)

    conv7 = slim.conv2d(concat3_6, scale*16, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv7_1'%name)
    conv7 = slim.conv2d(conv7, scale*16, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv7_2' % name)
    concat2_7 = deconv_and_concat(conv7, conv2)

    conv8 = slim.conv2d(concat2_7, scale*8, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv8_1'%name)
    conv8 = slim.conv2d(conv8, scale*8, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv8_2' % name)
    concat1_8 = deconv_and_concat(conv8, conv1)

    conv9 = slim.conv2d(concat1_8, scale*4, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv9_1'%name)
    conv9 = slim.conv2d(conv9, 3, [3, 3], activation_fn=tf.nn.relu, scope='%s/g_conv9_2' % name)

    return conv9