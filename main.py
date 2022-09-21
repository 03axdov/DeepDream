import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import PIL.Image as Image

from utils import *


def main():

    URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

    original_img = download(URL, max_dim=500)

    show(original_img)


if __name__ == "__main__":
    main()