import tensorflow as tf


def inception_v3(include_top=False):
    return tf.keras.applications.InceptionV3(include_top=include_top, weights="imagenet")


def dream_model(inputs, outputs):
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32)
        )
    )
    def __call__(self, img, steps, step_size):
        pass