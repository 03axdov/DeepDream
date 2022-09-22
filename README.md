# DeepDream
An implementation of DeepDream, essentially visualizing what triggers neural networks, in this case the pretrained inception_v3 model, trained on the imagenet dataset.
DeepDream does this by instead of subtracting features by their gradient*learning_rate, adding gradient*learning_rate in order to get features that 'trigger' the model as much as possible.
This results in a 'dream-like' image as seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62298758/191788958-58a42477-1713-497e-8797-3492ee30f2f3.jpg" height="auto" width="500" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/62298758/191789376-13832e11-12dd-47d7-8e60-2e1a9e0eabf3.jpg" height="auto" width="500" />
</p>

A simple implementation of DeepDream ^

<a href="https://www.tensorflow.org/tutorials/generative/deepdream">Tensorflow Tutorial</a>
