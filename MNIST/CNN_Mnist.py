import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(456917453838)

def load_data():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return(x_train, y_train), (x_test, y_test), class_names

def analyze_data(x_train, y_train, x_test, y_test, classes):
    print(50*'*')
    print("Training set shape:", x_train.shape, "and testing set shape:", x_test.shape)
    print("Training labels shape:", y_train.shape, "and testing labels shape:", y_test.shape)
    print("There are", x_train.shape[0], "elements in the training set and", x_test.shape[0], "elements in the testing set.")
    print("Example of training sample 8 label:", y_train[7])
    print(50*'*')

def visualize_data(x_train, y_train, classes):
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]), cmap='gray')
        plt.xlabel(classes[y_train[i]])
    #plt.show()

def prepare_data():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    analyze_data(x_train, y_train, x_test, y_test, classes)
    visualize_data(x_train, y_train, classes)
    #normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, classes

def create_cnn(num_classes):
    model = tf.keras.Sequential()
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same',
    activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    #output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #printing a summary of the model structure
    model.summary()
    return model

def compile_and_fit(model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation):

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])
    
    if not apply_data_augmentation:
        print("No data augmentation")
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                             validation_data=(x_test, y_test), shuffle=True)
    else:
        print("Using data augmentation")
        datagen=ImageDataGenerator(rotation_range=90, zoom_range=0, horizontal_flip=False,
                                    vertical_flip=True, rescale=None, preprocessing_function=None)
        datagen.fit(x_train)
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                       epochs=epochs, validation_data=(x_test, y_test), workers=1)
    return model, history

num_classes = 10
batch_size = 128
epochs = 3
apply_data_augmentation = True

x_train, y_train, x_test, y_test, classes = prepare_data()
cnn_model = create_cnn(num_classes)
cnn_model, history = compile_and_fit(cnn_model, x_train, y_train, x_test, y_test,
                                      batch_size, epochs, apply_data_augmentation)

score = cnn_model.evaluate(x_test, y_test)
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