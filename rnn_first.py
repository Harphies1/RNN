import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.optimizers import Adam

# GET THE DATA\
mnist = keras.datasets.mnist
# split the training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize the data
x_train = x_train/255.0
x_train = x_train/255.0
# print(x_train.shape[1:])
# print(x_train[0].shape)
# define the model
model = Sequential()
# add the layers
# Input layer
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu',
               return_sequences=True))
model.add(Dropout(0.2))

# add another layer
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

# add a dense layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# add ouput layer
model.add(Dense(10, activation='softmax'))

# define optimizer
optimize = Adam(lr=1e-3, decay=1e-5)
# compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimize,
              metrics=['accuracy'])
# fit the model
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
