# DeepDream
An implementation of DeepDream, essentially visualizing what triggers neural networks, in this case the pretrained inception_v3 model, trained on the imagenet dataset.
DeepDream does this by instead of subtracting features by their gradient * learning_rate, adding gradient * learning_rate in order to get features that "trigger" the model as much as possible.
This results in a "dream-like" image as seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62298758/191788958-58a42477-1713-497e-8797-3492ee30f2f3.jpg" height="auto" width="500" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/62298758/191966158-b8558eb6-7799-4ccf-9ec8-4b82cd98a3b9.jpg" height="auto" width="500" />
</p>

A simple implementation of DeepDream ^

<a href="https://www.tensorflow.org/tutorials/generative/deepdream">Tensorflow Tutorial</a>
