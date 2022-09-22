import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import PIL.Image as Image

from utils import download, show, run_deep_dream_simple
from models import inception_v3, dream_model, DeepDream


def main():

    URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

    # img = tf.keras.utils.get_file("original", origin=URL)
    # img = Image.open(img)
    # img.save("images/original.jpg")

    original_img = download(URL, max_dim=500)
    show(original_img)

    base_model = inception_v3()

    names=["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    model = dream_model(inputs=base_model.input, outputs=layers)
    deepdream = DeepDream(model)

    # tic = time.time()
    # dream_img = run_deep_dream_simple(original_img, deepdream, steps=100, learning_rate=0.01)
    # toc = time.time()
    # print(f"Time: {toc-tic} seconds")

    tic = time.time()
    OCTAVE_SCALE = 1.30

    img = tf.constant(np.array(original_img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)

        img = tf.image.resize(img, new_shape).numpy()

        img = run_deep_dream_simple(img, deepdream, steps=50, learning_rate=0.01)

    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img/255.0, tf.uint8)
    show(img)

    toc = time.time()
    print(f"Time: {toc-tic} seconds")

    # dream_img = Image.fromarray(img.numpy())
    # dream_img.save("images/dream_img.jpg")


if __name__ == "__main__":
    main()