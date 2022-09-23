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


def run_deep_dream_simple(img, deepdream, steps=100, learning_rate=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    learning_rate = tf.convert_to_tensor(learning_rate)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps=tf.constant(100)
        else:
            run_steps=tf.constant(steps_remaining)

        steps_remaining -= run_steps
        steps += run_steps

        loss, img = deepdream(img, steps, tf.constant(learning_rate))
        print(f"Step: {step}, Loss: {loss}")

    result = deprocess(img)
    show(result)

    return result


def random_roll(img, maxroll):
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled


def run_deep_dream_with_octaves(img, module, steps_per_octave=100, learning_rate=0.01, 
                                octaves=range(-2, 3), octave_scale=1.3,):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale**octave)
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = module(img, new_size)
            img = img + gradients*learning_rate

            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                print(f"Octave: {octave}, Step: {step}")

    result = deprocess(img)
    return result