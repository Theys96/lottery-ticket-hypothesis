#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies (not all may be necessary)
import sys

if len(sys.argv) < 3:
	print('usage: LotteryTicket.py PRUNING_FACTOR NUM_EPOCHS')
	sys.exit()
else:
	PRUNING_FACTOR = float(sys.argv[1])
	EPOCHS         = int(sys.argv[2])

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import matplotlib.pyplot as plt
import gzip
import tensorflow as tf
import tempfile
import zipfile
import os
import datetime
import json
# get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


#EPOCHS       = 100     # Set through command line
STEPS_EPOCH  = 100
num_steps    = EPOCHS * STEPS_EPOCH

PRUNING_START  = np.floor(.2*num_steps)
PRUNING_END    = np.floor(.6*num_steps)
PRUNING_FREQ   = np.floor(.02*num_steps)
#PRUNING_FACTOR = .01                      # Set through command line
goal_sparsity  = 1-PRUNING_FACTOR

RUN_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_ID   = RUN_DATE + "-" + str(PRUNING_FACTOR) + "-" + str(EPOCHS)
HIST_DIR = 'logs/hist/' + str(PRUNING_FACTOR) + "-" + str(EPOCHS) + '/'
if not os.path.exists(HIST_DIR):
    os.makedirs(HIST_DIR)


print("----- LOTTERY TICKET EXPERIMENT -----")
print("----------- CONFIGURATION: ----------")
print( f"\tNumber of epochs: {EPOCHS}" )
print( f"\tPruning from epoch {PRUNING_START/STEPS_EPOCH} to {PRUNING_END/STEPS_EPOCH} every {PRUNING_FREQ/STEPS_EPOCH} epochs")
print( f"\tTarget sparsity={goal_sparsity}, density={PRUNING_FACTOR}")
print( f"\tLogging directory: logs/fit/{RUN_ID}")
print("-------------------------------------")

# In[3]:


def read_mnist(images_path: str, labels_path: str):
	with gzip.open(labels_path, 'rb') as labelsFile:
		labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)
	with gzip.open(images_path,'rb') as imagesFile:
		length = len(labels)
		# Load flat 28x28 px images (784 px), and convert them to 28x28 px
		features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16)                         .reshape(length, 784)                         .reshape(length, 28, 28, 1)
	return features, labels

def display_image(position):
	image = train['features'][position].squeeze()
	plt.title('Example %d. Label: %d' % (position, train['labels'][position]))
	plt.imshow(image, cmap=plt.cm.gray_r)

train = {}
train['features'], train['labels'] = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

test = {}
test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


# In[4]:


def get_layer_density(lyr):
	mask = lyr._non_trainable_weights[0].numpy().flatten()
	return np.count_nonzero(mask) / len(mask)

def get_model_density(weights):
	mask1 = weights[6].flatten()
	mask2 = weights[11].flatten()
	return (np.count_nonzero(mask1) + np.count_nonzero(mask2))/(len(mask1) + len(mask2))

def setup_model(sparsity_to, sparsity_from=0, weights=None):
	pruning_params = {
	  'pruning_schedule': sparsity.PolynomialDecay(
		  initial_sparsity = sparsity_from,
		  final_sparsity   = sparsity_to,
		  begin_step = PRUNING_START,
		  end_step   = PRUNING_END,
		  frequency  = PRUNING_FREQ)
	}
	
	m = tf.keras.Sequential([
		tf.keras.layers.Conv2D(input_shape=(28,28,1),filters=6,kernel_size=(5,5),padding="same", activation="relu"),
		tf.keras.layers.MaxPool2D(pool_size=(2,2)),
		tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding="valid", activation="relu"),
		tf.keras.layers.MaxPool2D(pool_size=(2,2)),
		tf.keras.layers.Flatten(),
		sparsity.prune_low_magnitude( tf.keras.layers.Dense(units=120,activation="relu"), **pruning_params),
		sparsity.prune_low_magnitude( tf.keras.layers.Dense(units=84,activation="relu"), **pruning_params),
		tf.keras.layers.Dense(units=10, activation="softmax")])
	
	m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
	
	if weights is not None:
		m.set_weights(weights)
	
	return m

def train_model(m, name, earlyStop=True):
	#checkpoint = ModelCheckpoint("logs/checkpoints/" + name + ".h5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
	pruning = sparsity.UpdatePruningStep()
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + RUN_ID + "/" + name, histogram_freq=1)
	
	if earlyStop:
		callbcks = [tensorboard_callback,pruning,early]
	else:
		callbcks = [tensorboard_callback,pruning]
	
	hst = m.fit(
		x=train['features'], y=tf.keras.utils.to_categorical(train['labels'], 10), 
		validation_data=(test['features'], tf.keras.utils.to_categorical(test['labels'],10) ), 
		epochs=EPOCHS, steps_per_epoch=STEPS_EPOCH, 
		callbacks=callbcks)

	with open(HIST_DIR + name + '-' + RUN_DATE + '.json', 'w') as f:
		f.write( str(hst.history).replace('\'','"') )
		f.close()

	return hst


# In[5]:


# Pruning down to 10%
model1 = setup_model(goal_sparsity)
model1.summary()


# In[6]:


# Save random initialized weights
original_weights = np.copy(model1.get_weights())
print( "Original density is %.2f%%" % (get_model_density(original_weights)*100) )


# In[7]:


# Learn
hist1 = train_model(model1, "pruning_network", False) # Not allowed to early stop


# In[8]:


learnt_weights = np.copy(model1.get_weights())
print( "New density is %.2f%%" % (get_model_density(model1.get_weights())*100) )


# In[9]:


# set up lottery ticket: original weights but with the learnt pruning mask
lottery_ticket_weights     = np.copy(original_weights)
lottery_ticket_weights[6]  = np.copy(learnt_weights[6])  # pruning mask for layer 6
lottery_ticket_weights[11] = np.copy(learnt_weights[11]) # pruning mask for layer 7

# Setup lottery ticket model with initial sparsity of 90% 
model2 = setup_model(goal_sparsity, sparsity_from=goal_sparsity, weights=np.copy(lottery_ticket_weights) )

print( "Lottery ticket density is %.2f%%" % (get_model_density(model2.get_weights())*100) )


# In[10]:


# Train the lottery ticket
hist2 = train_model(model2, "lottery_ticket")


# In[11]:


# set up random weights, same architecture: new random weights but with the learnt pruning mask
model3 = setup_model(goal_sparsity, sparsity_from=goal_sparsity)

# extract new weights, inject pruning mask, put it back
model3_weights     = model3.get_weights()
model3_weights[6]  = np.copy(learnt_weights[6])
model3_weights[11] = np.copy(learnt_weights[11])
model3.set_weights(np.copy(model3_weights))


# In[ ]:


# Train the random reinitialization
hist3 = train_model(model3, "random_reinit")


# 
# # Extra
# 

# In[ ]:





# In[ ]:





# In[ ]:




