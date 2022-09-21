import tensorflow as tf
import PIL.Image as Image
import numpy as np

def download(URL, max_dim=None):
    name = URL.split("/")[-1]
    image_path = tf.keras.utils.get_file(name, origin=URL)
    img = Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))

    return np.array(img)


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def show(img):
    Image.fromarray(np.array(img)).show()


def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations=[layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)