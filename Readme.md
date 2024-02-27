
# Using and comparing Models for the translation of LSA signs

This project has the objetive of comparing different models and search for the one that's more effective for the translation of signs in the
Argentinian Sign Language. The models will be trained using Facundo Quiroga's dataset from the article "LSA64: A Dataset for Argentinian Sign Language"
http://facundoq.github.io/datasets/lsa64/

The main focus of this investigation relies of two types of networks:
- A convolutional network with LSTMs gate embeded in their units (ConvLSTM)
- A convolutional netowrk with LSTM layers at the end of their arquitecture (LRCN)

### Prepare the dataset

To prepare the dataset, download the videos and save them in a folder in the root of this project at './pruebas/all'.
Then excecute the following command:

```python dataset.py```

### Train the models

- To train the ConvLSTM model run:
    ```python model.py --model conv_lstm --name [name of the model]```

- To train the LRCN model run:
    ```python model.py --model lrcn --name [name of the model]```

### Preprocess experiment videos and live implementation

In this project there was also experimenting done on the input videos. Mainly, videos were transformed to only show visual information about the hands
through Mediapipe in the videos over a black background. 

There were two approaches taken in this project:
- The images of the hands areas were cut and put over a black background for every frame. Tu run this approach on videos on './pruebas/serie_prueba_sinGuantes'
run ```python main.py``` and for run it in a live translation run ```python main.py --live```
- The landmarks with skeletal information was drawn over a black background for every frame
Tu run this approach on videos on './pruebas/serie_prueba_sinGuantes' run ```python main.py``` and for run it in a live translation run ```python main.py --live```

### Tests the models

You can run experiments on ```experiments.ipynb```. Please follow the notebook for further information

### Augment data

To augment the data with different transformations run
```python dataAugmentation.py```

### Get dataset data into a json

Dataset labels are depicted in dataset.json. You can regenerate this file running
```python.py scrapper```
It will scrap the dataset labels from the original website.