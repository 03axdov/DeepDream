import tensorflow as tf
from utils import calc_loss


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
    def __call__(self, img, steps, learning_rate):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = img + gradients*learning_rate
            img = tf.clip_by_value(img, -1, 1)

        return loss, img