
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from TAE import *
from GRU_TAE import *
from random import randint
import copy

# Parameters that can be adapted
validation_size = 0.1
LEARNING_RATE = 0.01
BATCH_SIZE = 512
EPOCHS = 200
NB_DATA = 10000
LENGTH_PRED = 130 # max 200
STEP_PRED = 1
NB_TRAJ_TRAIN = 1800 # max 1800
add_noise = True

def load_training_data():
    saved_data = pickle.load(open('simulated_dataset.pkl', 'rb'))

    pos = list(map(lambda x:[t[0:3] for t in x[:-1]], saved_data))
    t = list(map(lambda x:[t[3] for t in x[:-1]], saved_data))
    X = saved_data[:,:,0:3]
    X = np.reshape(X, (X.shape + (1,)))
    print(np.shape(X))
    return pos, t, X

def split_train_val(pos, t, X, val_size):
    nb_train = int(round(len(X)*(1-val_size)))
    x_train = X[0:nb_train]
    t_train = t[0:nb_train]
    pos_train = pos[0:nb_train]
    x_test = X[nb_train:]
    t_test = t[nb_train:]
    pos_test = pos[nb_train:]

    return x_train, t_train, pos_train, x_test, t_test, pos_test


def add_noise(X):
    noise = np.random.normal(0,0.002, size=np.shape(X))
    noise = np.clip(noise, -0.01, 0.01)
    X += noise
    return X

def create_sparse_data(x_train, nb_data=10):
    x_train_sparse = []
    y_train_sparse = []
    for i in range(nb_data):
        num_traj = randint(0,np.shape(x_train)[0]-1)
        time_measured = randint(20,200)

        new_data = copy.deepcopy(x_train[num_traj])
        y_train_sparse.append(copy.deepcopy(new_data))
        new_data[time_measured:200] = 0
        vect_obs = np.ones((np.shape(new_data)[0],1,1))
        vect_obs[time_measured:200] = 0
        new_data = np.concatenate((new_data, vect_obs), axis=1)
        x_train_sparse.append(copy.deepcopy(new_data))

    x_train_sparse = np.array(x_train_sparse)
    y_train_sparse = np.array(y_train_sparse)

    return x_train_sparse, y_train_sparse

def trunc_data(data, nb_obs=30):
    new_data = copy.deepcopy(data)
    new_data[nb_obs:200] = 0
    vect_obs = np.ones((np.shape(new_data)[0],1,1))
    vect_obs[nb_obs:200] = 0

    new_data = np.concatenate((new_data, vect_obs), axis=1)
    new_data = np.reshape(new_data, (1,200,4,1))

    return new_data

def test_autoencoder(x_test, autoencoder, nb_obs=60):
    x_test_sparse = []
    for traj in x_test:
        new_data = copy.deepcopy(traj)
        new_data[nb_obs:200] = 0
        vect_obs = np.ones((np.shape(new_data)[0],1,1))
        vect_obs[nb_obs:200] = 0
        new_data = np.concatenate((new_data, vect_obs), axis=1)
        x_test_sparse.append(copy.deepcopy(new_data))
        
    x_test_sparse = np.array(x_test_sparse)
    predictions = autoencoder.predict(x_test_sparse)
    return predictions

def train_autoencoder(x_sparse, x_train, learning_rate, batch_size, epochs, length_pred):
# Change in function of the architecture used for the prediction
    autoencoder = TAE(input_shape=(length_pred,4,1), output_shape=(length_pred,3,1), latent_space_dim=2)
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    print(np.shape(x_sparse), np.shape(x_train))
    autoencoder.train(x_sparse, x_train, batch_size, epochs)
    return autoencoder

if __name__ == "__main__":
    pos_dataset, t_dataset, X_dataset = load_training_data()
    if add_noise == True:
        X_dataset = add_noise(X_dataset)
    x_train, t_train, pos_train, x_test, t_test, pos_test = split_train_val(pos_dataset, t_dataset, X_dataset, validation_size)
    x_train = x_train[0:NB_TRAJ_TRAIN,:]
    x_sparse, y_sparse = create_sparse_data(x_train,NB_DATA)

    autoencoder = train_autoencoder(x_sparse[:,0:LENGTH_PRED:STEP_PRED], y_sparse[:,0:LENGTH_PRED:STEP_PRED], LEARNING_RATE, BATCH_SIZE, EPOCHS, int(LENGTH_PRED/STEP_PRED))
    results = test_autoencoder(x_test[:,0:LENGTH_PRED:STEP_PRED], autoencoder)
    results = results[0]
