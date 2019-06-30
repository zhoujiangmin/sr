from model import *
import tensorflow as tf
import h5py
import numpy as np
import random

def get_batch_data(idx_shuf, iter):
    _input = input_data[iter*batch_size:(iter+1)*batch_size, :, :, :]
    _target = target_data[iter * batch_size:(iter + 1) * batch_size, :, :, :]
    return _input, _target

def train():
    input = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3], name='input')
    target = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3], name='target')
    output = UNet(input, scale=24, name='U-Net')
    mse = tf.reduce_sum(tf.abs(output-target, name='abs'))/(batch_size*input_shape*input_shape*3)

    saver = tf.train.Saver()
    train_op = tf.train.AdamOptimizer(1e-4).minimize(mse)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for ep in range(epoch):
            idx_shuf = list(range(data_num))
            random.shuffle(idx_shuf)
            for iter in range(iter_num):
                _input, _target = get_batch_data(idx_shuf, iter)
                feed_dict = {input:_input, target:_target}
                _mse, _ = sess.run([mse, train_op], feed_dict=feed_dict)
                if iter%10 == 0:
                    print("epoch: %d, iter: %d, mse:%f"%(ep, iter, _mse))

            print("===>save ckpt_%d"%ep)
            saver.save(sess, "ckpt/model_%d"%ep)

if __name__ == '__main__':
    input_shape = 64
    epoch = 100
    batch_size = 24

    f = h5py.File('D:\project\sr\\train_data\\v1\\train_10.h5', 'r')
    input_data = f['input'][:]/255
    target_data = f['target'][:]/255
    data_num = input_data.shape[0]
    iter_num = data_num//batch_size

    train()