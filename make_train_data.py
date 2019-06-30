import numpy as np
import argparse
import glob
import os
import cv2
import h5py

def cut_patches_from_one_image(image_file, file_index):
    patch_index = file_index*num_per_image
    img_input = []
    img_target = []
    img = cv2.imread(image_file)
    for i in range(num_per_image):
        h, w, _ = img.shape
        x = np.random.randint(0, h-patch_size)
        y = np.random.randint(0, w-patch_size)
        patch_target = img[x:x+patch_size, y:y+patch_size, :]
        patch_resize = patch_target.copy()
        patch_resize = cv2.resize(patch_resize, (patch_size//2, patch_size//2), interpolation=cv2.INTER_AREA)
        patch_resize = cv2.resize(patch_resize, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        # cv2.imwrite("patch_org_%d.png"%i, patch)
        # cv2.imwrite("patch_resize_%d.png"%i, patch_resize)
        img_input.append(patch_resize)
        img_target.append(patch_target)
    return img_input, img_target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', dest='image_path', type=str, default='D:\d7100', help='the source image path')
    parser.add_argument('--data_path', dest='data_path', type=str, default='D:\project\sr\\train_data\\v1')
    parser.add_argument('--file_name', dest='file_name', type=str, default='train')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=64, help="the size of square patch")
    parser.add_argument('--num_per_image', dest='num_per_image', type=int, default=10, help="the number of patches")
    args = parser.parse_args()

    image_path = args.image_path
    data_path = args.data_path
    file_name = args.file_name
    patch_size = args.patch_size
    num_per_image = args.num_per_image
    file_name = "%s_%s.h5"%(file_name, str(num_per_image))

    file_list = glob.glob(os.path.join(image_path, '*.jpg'))
    train_input = []
    train_target = []
    for idx, f in enumerate(file_list[:100]):
        img_input, img_target = cut_patches_from_one_image(f, idx)
        train_input += img_input
        train_target += img_target

    f = h5py.File(os.path.join(data_path, file_name), 'w')
    f['input'] = np.array(train_input)
    f['target'] = np.array(train_target)
    f.close()




