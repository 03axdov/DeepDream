import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import PIL.Image as Image

from utils import download, show
from models import inception_v3, dream_model


def main():

    URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

    original_img = download(URL, max_dim=500)
    show(original_img)

    base_model = inception_v3()

    names=["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).ouput for name in names]

    dream_model = dream_model(inputs=base_model.input, outputs=layers)


if __name__ == "__main__":
    main()