from keras.models import Sequential,load_model
from keras.layers import ConvLSTM2D, Flatten, MaxPooling3D,TimeDistributed,Dropout, Conv2D,MaxPooling2D, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
#from keras.applications import EfficientNetB0
#from keras.utils import to_categorical
import cv2
import numpy as np
import json
import pathlib
import random
import os

import argparse

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

HEIGHT = 224
WIDTH = 224
N_FRAMES = 20
num_classes = 64

parser = argparse.ArgumentParser()
parser.add_argument("--model", action='store', help="Determines the model type, conv_lstm or lrcn")
parser.add_argument("--name",action="store",help="Determines name of the model. If the model exists it will load it and keep training")

if parser.name == None:
   raise Exception("Please specify a name")

nameModel = parser.name

subset_paths = {
  'train': pathlib.Path('./dataset/train'),
  'val': pathlib.Path('./dataset/val'),
  'test': pathlib.Path('./dataset/test')
}

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def LRCN():
    model = Sequential()
    # Capas convolucionales para procesar cada cuadro de video
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=(N_FRAMES,WIDTH, HEIGHT, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))

    # Capas LSTM para manejar la secuencialidad temporal
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))

    # Capa de salida con neuronas igual al número de clases de lengua de señas
    model.add(Dense(num_classes, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

""" def EffLRCN():
    model = Sequential()
    # Capas convolucionales para procesar cada cuadro de video
    model.add(EfficientNetB0(input_shape=(N_FRAMES,WIDTH, HEIGHT),pooling=(4,4),classes=num_classes,include_top = False))
    # Capas LSTM para manejar la secuencialidad temporal
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))

    # Capa de salida con neuronas igual al número de clases de lengua de señas
    model.add(Dense(num_classes, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model """

def convLSTM():
  model = Sequential()
  model.add(ConvLSTM2D(filters=4,kernel_size=(3,3), activation='tanh', data_format='channels_last',
                       recurrent_dropout=0.2,return_sequences=True, input_shape = (N_FRAMES,
                                                                                   WIDTH, HEIGHT,3)))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last'))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=8,kernel_size=(3,3), activation='tanh', data_format='channels_last',
                       recurrent_dropout=0.2,return_sequences=True))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last'))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=14,kernel_size=(3,3), activation='tanh', data_format='channels_last',
                       recurrent_dropout=0.2,return_sequences=True))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last'))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=16,kernel_size=(3,3), activation='tanh', data_format='channels_last',
                       recurrent_dropout=0.2,return_sequences=True))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last'))

  model.add(Flatten())

  model.add(Dense(num_classes,activation='softmax'))
  return model
  

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
  #print('Procesando video:',video_path)
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
  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  frame = np.expand_dims(frame, axis=-1) 
  result.append(frame/255)
  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
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

#input_shape = (cant_videos,cant_frames,img_size)

def find_id(labels,name):
   return list(filter(lambda x: x['name'] == name,labels))[0]

def get_files_and_class_names(path):
    video_paths = list(path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    f = open('./dataset.json')
    labels = json.load(f)
    classes = np.array(list(map(lambda x: int(find_id(labels,x)['id'])-1,classes)))
    videos = np.array([frames_extraction(str(p)) for p in video_paths ],dtype='float16')
    return videos, classes

print('Preparando datos...')

with open('x_train_full.npy','wb') as f:
   np.save(f, get_files_and_class_names(subset_paths['train']))

x_train, y_train = get_files_and_class_names(subset_paths['train'])
X_val, y_val = get_files_and_class_names(subset_paths['val'])

#x_train = np.load('x_train_full.npy')
#y_train = np.load('y_train_full.npy')
#X_val = np.load('x_val_full.npy')
#y_val = np.load('y_val_full.npy')

print('shape x_train:', x_train.shape)
print('Empezando creacion de modelo y entrenamiento...')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights= True)
checkpoint_callback = ModelCheckpoint(nameModel,save_best_only=True,monitor='val_loss',mode='min')

train_gen = DataGenerator(x_train, y_train, 8)
val_gen = DataGenerator(X_val, y_val, 8)

if os.path.exists(f'./models/{nameModel}'):
   print('Loading saved model...')
   model = load_model(f'./models/{nameModel}')
   model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val),callbacks=[early_stopping_callback,checkpoint_callback])
   model.save(f'./models/{nameModel}')
else:
  if parser.model == 'conv_lstm':
    model = convLSTM()
  elif parser.model == 'lrcn':
    model = LRCN()
  else:
    raise Exception("Model not found, choose conv_lstm or lrcn")
  model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  model.summary()
  model.fit(train_gen, epochs=60, batch_size=8, validation_data=val_gen,callbacks=[early_stopping_callback,checkpoint_callback])

  model.save(f'./models/{nameModel}')