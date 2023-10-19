import cv2
import numpy as np
import json
import pathlib
import random
from keras.utils import to_categorical

subset_paths = {
  'train': pathlib.Path('./dataset/train'),
  'val': pathlib.Path('./dataset/val'),
  'test': pathlib.Path('./dataset/test')
}

HEIGHT = 224
WIDTH = 224
N_FRAMES = 20
num_classes = 64

def groupFrames(video_path, n_frames = N_FRAMES, frame_step = 15):
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
  src = cv2.VideoCapture(video_path)  

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
  print(f'LEYO BIEN EL FRAME:{ret}')
  if ret:
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      frame = np.expand_dims(frame, axis=-1) 
      result.append(frame/255)
  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    print(f'LEYO BIEN EL FRAME:{ret}')
    if ret:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1) 
        result.append(frame/255)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)
  #print('SHAPE RESULT',result.shape)
  return result

def find_id(labels,name):
   return list(filter(lambda x: x['name'] == name,labels))[0]

def get_files_and_class_names(path):
    video_paths = list(path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    f = open('./dataset.json')
    labels = json.load(f)
    classes = np.array(list(map(lambda x: int(find_id(labels,x)['id'])-1,classes)))
    encoded_labels = to_categorical(classes)
    videos = np.array([groupFrames(str(p)) for p in video_paths ])
    return videos, classes


x_test, y_test = get_files_and_class_names(subset_paths['test'])