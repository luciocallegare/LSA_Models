import random
import pathlib


import cv2
import einops
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

subset_paths = {
  'train': pathlib.Path('./dataset2/train'),
  'val': pathlib.Path('./dataset2/val'),
  'test': pathlib.Path('./dataset2/test')
}

n_frames = 20
batch_size = 8
HEIGHT = 224
WIDTH = 224


def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  frame = frame/255
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (WIDTH,HEIGHT), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
  need_length = 1 + (n_frames - 1) * frame_step
  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)
  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  #frame = np.expand_dims(frame, axis=-1) 
  result.append(format_frames(frame, output_size))
  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      #frame = format_frames(frame, output_size)
      #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      #frame = np.expand_dims(frame, axis=-1) 
      result.append(format_frames(frame, output_size))
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]
  #print('SHAPE RESULT',result.shape)
  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()
    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames) 
      #print(name, path)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

def main():
  output_signature = (tf.TensorSpec(shape = (None, None, None,3), dtype = tf.float32),
                      tf.TensorSpec(shape = (), dtype = tf.int16))
  val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], n_frames),
                                          output_signature = output_signature)
  val_ds = val_ds.batch(batch_size)
  train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'],n_frames,training=True),
                                            output_signature = output_signature)
  train_ds = train_ds.batch(batch_size)
  test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], n_frames),
                                          output_signature = output_signature)
  test_ds = test_ds.batch(batch_size)

  input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
  input = layers.Input(shape=(input_shape[1:]))
  x = input

  x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

  # Block 1
  x = add_residual_block(x, 16, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

  # Block 2
  x = add_residual_block(x, 32, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

  # Block 3
  x = add_residual_block(x, 64, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

  # Block 4
  x = add_residual_block(x, 128, (3, 3, 3))

  x = layers.GlobalAveragePooling3D()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64)(x)

  model = keras.Model(input, x)

  frames, label = next(iter(train_ds))
  model.build(frames)
  model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                optimizer = keras.optimizers.Adam(learning_rate = 0.001), #original 0.0001
                metrics = ['accuracy'])
  history = model.fit(x = train_ds,
                      epochs = 50, 
                      validation_data = val_ds)

  model.save("lsa_rna")


main()