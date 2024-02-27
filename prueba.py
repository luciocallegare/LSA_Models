import cv2
import numpy as np
import json
import pathlib
import random

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
  print(f'Video url: {video_path}, video_length:{video_length.decode("utf-8")}')
  need_length = 1 + (n_frames - 1) * frame_step
  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)
  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  if ret:
      cv2.imshow('prueba',frame)  
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      frame = np.expand_dims(frame, axis=-1) 
      result.append(frame/255)
  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
        cv2.imshow('prueba',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1) 
        result.append(frame/255)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  cv2.destroyAllWindows()
  result = np.array(result)
  #print('SHAPE RESULT',result.shape)
  return result


def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''
    encoded_string = video_path.encode('latin-1')
    video_path = encoded_string.decode("utf-8")
    
    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video url: {video_path}, video_length:{video_frames_count}')
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/N_FRAMES), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(N_FRAMES):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break
            
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #frame = np.expand_dims(frame, axis=-1) 

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    # Release the VideoCapture object. 
    video_reader.release()

    # Return the frames list.
    return frames_list


def find_id(labels,name):
   return list(filter(lambda x: x['name'] == name,labels))[0]

def get_files_and_class_names(path):
    video_paths = list(path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    f = open('./dataset.json')
    labels = json.load(f)
    classes = np.array(list(map(lambda x: int(find_id(labels,x)['id'])-1,classes)))
    videos = np.array([frames_extraction(str(p)) for p in video_paths ])
    return videos, classes


x_test, y_test = get_files_and_class_names(subset_paths['test'])