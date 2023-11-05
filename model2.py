from keras.models import Sequential,load_model
from keras.layers import ConvLSTM2D, Flatten, MaxPooling3D,TimeDistributed,Dropout, Conv2D,MaxPooling2D, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import EfficientNetB0
from keras.utils import to_categorical
import cv2
import numpy as np
import json
import pathlib
import random
import os

HEIGHT = 224
WIDTH = 224
N_FRAMES = 20
num_classes = 64

nameModel = 'modelFullVids'

subset_paths = {
  'train': pathlib.Path('./dataset_full/train'),
  'val': pathlib.Path('./dataset_full/val'),
  'test': pathlib.Path('./dataset_full/test')
}

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

def convLSTM():
  model = Sequential()
  model.add(ConvLSTM2D(filters=4,kernel_size=(3,3), activation='tanh', data_format='channels_last',
                       recurrent_dropout=0.2,return_sequences=True, input_shape = (N_FRAMES,
                                                                                   WIDTH, HEIGHT,1)))
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

#input_shape = (cant_videos,cant_frames,img_size)

def find_id(labels,name):
   encoded_string = name.encode('latin-1')
   name = encoded_string.decode("utf-8")
   return list(filter(lambda x: x['name'] == name,labels))[0]

def get_files_and_class_names(path):
    video_paths = list(path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    f = open('./dataset.json')
    labels = json.load(f)
    classes = np.array(list(map(lambda x: int(find_id(labels,x)['id'])-1,classes)))
    encoded_labels = to_categorical(classes)
    videos = np.array([groupFrames(str(p)) for p in video_paths ],dtype='float16')
    return videos, classes

def cant_frames_promedio(path):
    video_paths = list(path.glob('*/*.avi'))
    cant_frames = [len(groupFrames(str(p))) for p in video_paths ]
    print(cant_frames)
    return sum(cant_frames)/len(cant_frames)

print('Preparando datos...')
#x_train, y_train = get_files_and_class_names(subset_paths['train'])
#X_val, y_val = get_files_and_class_names(subset_paths['val'])

x_train = np.load('x_train_full.npy')
y_train = np.load('y_train_full.npy')
X_val = np.load('x_val_full.npy')
y_val = np.load('y_val_full.npy')

print('shape x_train:', x_train.shape)
print('Empezando creacion de modelo y entrenamiento...')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights= True)
checkpoint_callback = ModelCheckpoint(nameModel,save_best_only=True,monitor='val_loss',mode='min')

if os.path.exists(f'./{nameModel}'):
   print('Loading saved model...')
   model = load_model(f'./{nameModel}')
   model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val),callbacks=[early_stopping_callback,checkpoint_callback])
   model.save(nameModel)
else:
  model = convLSTM()
  model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  model.summary()
  model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val),callbacks=[early_stopping_callback,checkpoint_callback])

  model.save(nameModel)