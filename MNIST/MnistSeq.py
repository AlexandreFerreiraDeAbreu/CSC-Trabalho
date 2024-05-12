import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model

tf.random.set_seed(456917453838)
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[i], cmap="gray")
plt.show()

print(tf.version)

print('***** Log data shape *****')
print('Train data shape', x_train.shape)
print('Test data shape', x_test.shape)
print('Number of training samples', x_train.shape[0])
print('Number of testing samples', x_test.shape[0])
print('The labels...', y_test)
print('**************************')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255
input_shape = x_train.shape[1]
print(x_train.shape)
print(y_train.shape)
print(input_shape)

model = tf.keras.Sequential([
    layers.Dense(input_shape),
    layers.Dense(128 , activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=0.1)

predictions = model.predict(x_test[:10])

for i, prediction in enumerate(predictions):
    predicted_value = tf.argmax(prediction)
    predicted_label = class_names[predicted_value]
    real_label = class_names[y_test[i]]
    print('Predicted label: ', predicted_label, '|| Real label: ', real_label)