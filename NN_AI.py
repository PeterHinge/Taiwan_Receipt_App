import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


from FirstApp import app


mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


"""Uncomment the following to get a feel for the data"""
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(258, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)


"""Using test sample to check a for model over fitting."""
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss)
print(test_acc)


draw_exit = False
while not draw_exit:
    new_digit = app()

    for digit in new_digit:
        digit = [i/255 for i in digit]
        print("")
        print("")
        print("")
        print(digit)

        model_prediction = model.predict([[digit]])

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(digit, cmap=plt.cm.binary)

        plt.subplot(1, 2, 2)
        plt.xticks([i for i in range(10)])
        plt.bar(range(10), model_prediction[0][0:10], color="#777777")
        plt.ylim(0.0, 1.0)

        plt.show()

        """Makes predicted output to the terminal."""
        prediction = np.argmax(model_prediction)
        print("The prediction of the new digit is: {}".format(prediction))

    print("Do you wish to exit? (Y for yes, all other for no)")
    if input() == "Y":
        draw_exit = True
