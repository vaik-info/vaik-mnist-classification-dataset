import argparse
import os
import random
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def crop_char_image(mnist_char_image):
    x_min, y_min, x_max, y_max = 0, 0, mnist_char_image.shape[1], mnist_char_image.shape[0]

    x_sum_mnist_char_image = np.sum(mnist_char_image, axis=0)
    y_sum_mnist_char_image = np.sum(mnist_char_image, axis=1)

    for i in range(x_sum_mnist_char_image.shape[0]):
        if x_sum_mnist_char_image[i] > 0:
            x_min = i
            break

    for i in range(y_sum_mnist_char_image.shape[0]):
        if y_sum_mnist_char_image[i] > 0:
            y_min = i
            break

    x_revert_sum_diff_bool_image = x_sum_mnist_char_image[::-1]
    y_revert_sum_diff_bool_image = y_sum_mnist_char_image[::-1]

    for i in range(x_revert_sum_diff_bool_image.shape[0]):
        if x_revert_sum_diff_bool_image[i] > 0:
            x_max = x_revert_sum_diff_bool_image.shape[0] - i
            break

    for i in range(y_revert_sum_diff_bool_image.shape[0]):
        if y_revert_sum_diff_bool_image[i] > 0:
            y_max = y_revert_sum_diff_bool_image.shape[0] - i
            break

    return mnist_char_image[y_min:y_max, x_min:x_max]


def write(output_sub_dir_path, sample_num, char_max_size, char_min_size, x, y, classes):
    os.makedirs(output_sub_dir_path, exist_ok=True)

    for file_index in tqdm(range(sample_num), desc=f'write at {output_sub_dir_path}'):
        canvas = np.zeros(
            (random.randint(char_min_size, char_max_size), random.randint(char_min_size, char_max_size), 3),
            dtype=np.uint8)
        mnist_index = random.randint(0, y.shape[0] - 1)

        mnist_char_image = np.array(Image.fromarray(x[mnist_index]).resize((canvas.shape[1], canvas.shape[0])))
        mnist_char_image = np.clip((mnist_char_image > 125) * 255, 0, 255)
        for color_index in range(canvas.shape[2]):
            canvas[:, :, 0] = mnist_char_image * random.randint(0, 255)
            canvas[:, :, 1] = mnist_char_image * random.randint(0, 255)
            canvas[:, :, 2] = mnist_char_image * random.randint(0, 255)

        output_class_sub_dir_path = os.path.join(output_sub_dir_path, classes[y[mnist_index]])
        os.makedirs(output_class_sub_dir_path, exist_ok=True)
        file_name = f'{os.path.basename(os.path.splitext(output_sub_dir_path)[0])}_{file_index:09d}'
        output_image_path = os.path.join(output_class_sub_dir_path, f'{file_name}.jpg')
        Image.fromarray(canvas).save(output_image_path, quality=100, subsampling=0)


def main(output_dir_path, train_sample_num, valid_sample_num, char_max_size, char_min_size):
    os.makedirs(output_dir_path, exist_ok=True)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    classes_txt_path = os.path.join(os.path.dirname(__file__), 'classes.txt')
    classes = []
    with open(classes_txt_path) as f:
        for line in f:
            classes.append(line.strip())

    output_train_dir_path = os.path.join(output_dir_path, 'train')
    write(output_train_dir_path, train_sample_num, char_max_size, char_min_size, x_train, y_train, classes)

    output_valid_dir_path = os.path.join(output_dir_path, 'valid')
    write(output_valid_dir_path, valid_sample_num, char_max_size, char_min_size, x_test, y_test, classes)

    shutil.copy(classes_txt_path, os.path.join(output_dir_path, os.path.basename(classes_txt_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-mnist-classification-dataset')
    parser.add_argument('--train_sample_num', type=int, default=50000)
    parser.add_argument('--valid_sample_num', type=int, default=2000)
    parser.add_argument('--char_max_size', type=int, default=256)
    parser.add_argument('--char_min_size', type=int, default=160)
    args = parser.parse_args()

    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
