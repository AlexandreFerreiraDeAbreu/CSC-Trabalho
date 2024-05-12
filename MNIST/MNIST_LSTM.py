import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

tf.random.set_seed(91195003)
np.random.seed(91190530)
tf.keras.backend.clear_session()

def load_data():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    print(x_train.shape)
    print(x_test.shape)
    return x_train, y_train, x_test, y_test, class_names

def create_LSTM():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(28, 28), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    return model

def compile_and_fit(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                   optimizer = tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    return model, history

batch_size = 128
epochs = 3

x_train, y_train, x_test, y_test, classes = load_data()
LSTM_model = create_LSTM()
LSTM_model, history = compile_and_fit(LSTM_model, x_train, y_train, x_test, y_test,
                                      epochs, batch_size)
score = LSTM_model.evaluate(x_test, y_test)
print("Evaluation loss:", score[0])
print("Evaluation accuracy", score[1])

def plot_learning_curves(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    #creating figure
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training accuracy')
    plt.plot(epochs_range, val_acc, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training/Validation accuracy')
    plt.subplot(1,2,1)
    plt.plot(epochs_range, loss, label='Training loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='lower right')
    plt.title('Training/Validation loss')
    plt.show()

plot_learning_curves(history, epochs)



