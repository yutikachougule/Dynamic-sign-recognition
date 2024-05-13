import os
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from data_utils import labels_to_number, videos_to_dict
from frame_generator import VideoFrameGenerator
from defined_model import model_asl
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

height = 224
width = 224
dim = (height, width)
batch_size = 8
frames = 10
channels = 3
output = 11

TRAIN_PATH = './data/train/'
VAL_PATH = './data/val/'
TEST_PATH = './data/test/'

# transform labels from string to number
labels = labels_to_number(TRAIN_PATH)
# print(f'Labels: {labels}')

# load dataset as dict
y_train_dict = videos_to_dict(TRAIN_PATH, labels)
y_val_dict = videos_to_dict(VAL_PATH, labels)
y_test_dict = videos_to_dict(TEST_PATH, labels)

print(f'\nTrain set: {len(y_train_dict)} videos - with labels')
print(f'Val   set: {len(y_val_dict)} videos - with labels')
print(f'Test  set: {len(y_test_dict)} videos - with labels')
print(f'Train set samples: {list(islice(y_train_dict.items(), 3))}')
print(f'Val   set samples: {list(islice(y_val_dict.items(), 3))}')
print(f'Test  set samples: {list(islice(y_test_dict.items(), 3))}')

# get video paths (without labels)  
X_train = list(y_train_dict.keys())
X_val = list(y_val_dict.keys())
X_test = list(y_test_dict.keys())

print(f'\nTrain set: {len(X_train)} videos')
print(f'Val   set: {len(X_val)} videos')
print(f'Test  set: {len(X_test)} videos')
print(f'Train set samples: {X_train[:4]}')
print(f'Val   set samples: {X_val[:4]}')
print(f'Test  set samples: {X_test[:4]}')

# instantiation of generators for train and val sets
print('\nTrain generator')
train_generator = VideoFrameGenerator(
    list_IDs=X_train,
    labels=y_train_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=3,
    n_sequence=frames,
    shuffle=True,
    type_gen='train'
)

print('\nVal generator')
val_generator = VideoFrameGenerator(
    list_IDs=X_val,
    labels=y_val_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=3,
    n_sequence=frames,
    shuffle=True,
    type_gen='val'
)

# model building
print('\nModel building and compiling . . .')
model = model_asl(frames, width, height, channels, output)
model.summary()
# model compiling
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,amsgrad=False, name="Adam") 
model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

# callbacks creation
if not os.path.isdir('./saved_models/'):
    os.mkdir('./saved_models/')

# save the best model each time
# path = './saved_models/'
# checkpoint_cb = ModelCheckpoint(path + 'best_model_mo_300.keras', save_best_only=True)

best_model_path = './saved_models/best_model_mobilenet_10_300.keras'
every_100_epochs_path = './saved_models/model-{epoch:04d}.keras'

# Create a callback to save the best model
best_model_cb = ModelCheckpoint(
    filepath=best_model_path, 
    save_best_only=True, 
    verbose=1, 
    # save_weights_only=True,
   )  # or 'max' if the metric should be maximized

# Create a callback to save the model's weights every 100 epochs
epoch_checkpoint_cb = ModelCheckpoint(
    filepath=every_100_epochs_path, 
    verbose=1, 
    save_best_only=True,
    save_freq=100*len(train_generator)//batch_size)  # calculate the number of steps per epoch if using `steps_per_epoch`


# start training
print('\nStart training . . .')
learn_epochs = 300
history = model.fit(train_generator,validation_data=val_generator,epochs=learn_epochs,callbacks=[best_model_cb,epoch_checkpoint_cb])

# save learning curves
if not os.path.isdir('./plots/'):
    os.mkdir('./plots/')

print('\nSaving learning curves graph . . .')
pd.DataFrame(history.history).plot(figsize=(9, 6))
plt.grid(True)
plt.gca().set_ylim(0, 4)
plt.savefig('./plots/learning_curves.png')
plt.close() 

loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation loss
plt.figure(figsize=(9, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./plots/loss_plot.png')
plt.close() 


# Check if accuracy is part of the metrics
if 'accuracy' in history.history:
    # Extract training and validation accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Plot training and validation accuracy
    plt.figure(figsize=(9, 6))
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/accuracy_plot.png')
    plt.close()

print('\nLearning curves and individual plots for loss and accuracy have been saved.')
