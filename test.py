from model import *
import tensorflow as tf
import cv2
import numpy as np
import os

def test_block(image_block):
    block = tf.placeholder(tf.float32, [None, block_size, block_size, 3], name='block')
    output_img = UNet(block, scale=24, name='U-Net')
    variables = tf.contrib.framework.get_variables_to_restore()
    saver = tf.train.Saver(variables)
    print(variables)
    with tf.Session() as sess:
        saver.restore(sess, "ckpt/model_99")
        output_img = sess.run(output_img)
    return output_img

def test(image):
    block = tf.placeholder(tf.float32, [None, block_size, block_size, 3], name='block')
    output_img = UNet(block, scale=24, name='U-Net')
    variables = tf.contrib.framework.get_variables_to_restore()
    saver = tf.train.Saver(variables)
    print(variables)
    sess = tf.Session()
    saver.restore(sess, "ckpt/model_99")

    _, h, w, c = image.shape
    h_block_num = h//block_size
    w_block_num = w//block_size
    for i in range(h_block_num):
        for j in range(w_block_num):
            input_block = image[:, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :]
            #print(block.shape)
            output_block = sess.run([output_img], feed_dict={block:input_block})
            image[:, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :] = output_block
    return image


if __name__ == '__main__':
    input_shape = 64
    batch_size = 24
    block_size = 256
    img = 'D:/project/sr/test_set/1.jpg'
    path, file_name = os.path.split(img)
    filename, extend = os.path.splitext(file_name)
    resize_filename = file_name+'_resize'+ extend

    image = cv2.imread(img)
    h, w, c = image.shape
    image_resize = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path, resize_filename), image_resize)

    image_resize = test(np.array([image_resize/255]))[0]
    image_resize = image_resize*255
    resize_filename = file_name + '_tf_resize' + extend
    cv2.imwrite(os.path.join(path, resize_filename), image_resize)