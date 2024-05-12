import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

tf.random.set_seed(456917453838)

def load_large_dataset(image_size, batch_size):
    class_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    train_dataset, validation_data = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/david/Desktop/MMC/CSC/Trabalho/ASl/asl_alphabet_train',
                                                                labels='inferred', label_mode='categorical', class_names=class_names,
                                                                batch_size=batch_size, interpolation='nearest', image_size=image_size)
    return train_dataset, validation_data, 29

def load_small_dataset(image_size, batch_size):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
                    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                      'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    train_dataset, validation_data = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/david/Desktop/MMC/CSC/Trabalho/ASl/asl_dataset_pequeno',
                                                                labels='inferred', label_mode='categorical', batch_size=batch_size,
                                                                image_size=image_size, interpolation='nearest', validation_split=0.1,
                                                                subset='both', seed=456, class_names=class_names)
    #train_dataset = train_dataset.map(normalize)
    return train_dataset, validation_data, 36

#dataset = 2718 sets de 32 imagens, cada destes sets tem um tuplo com as 32 imagens à "esquerda" e as respetivas labels à "direita"
#elem[0][0] representa a primeira imagem
#elem[1][0] representa o label através de one-hot-encoding

def visualize_data(dataset):
    for elem in dataset.take(1):
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(elem[0][i])
        plt.show()

        print('***** Log data shape *****')
        print('Train data batch shape', elem[0].shape)
        print(elem[0][0])
        #print('Test data shape', x_test.shape)
        #print('Number of training samples', x_train.shape[0])
        #print('Number of testing samples', x_test.shape[0])
        #print('The labels...', y_test)
        print('**************************')

#visualize_data(load_small_dataset())

def model_create_compile_and_fit():
    image_size = (200, 200)
    batch_size = 32
    epochs = 15
    dataset, validation_data, output_shape = load_small_dataset(image_size, batch_size)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,3,activation='relu',padding='same',input_shape = (image_size[0], image_size[1], 3)))
    model.add(tf.keras.layers.Conv2D(32,3,activation='relu',padding='same'))
    #model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 2
    model.add(tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'))
    #model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))

    #Block 3
    model.add(tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'))
    #model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))
    model.add(tf.keras.layers.Dropout(0.4))

    # fully connected layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # output layer
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(dataset, epochs=epochs, validation_data=validation_data)

    score = model.evaluate(validation_data)
    print("Evaluation loss:", score[0])
    print("Evaluation accuracy", score[1])
    return history, epochs

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

history, epochs = model_create_compile_and_fit()
plot_learning_curves(history, epochs)