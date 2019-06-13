import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

WINNING_NUMS = []


def tf_model():

    data = np.genfromtxt('C:\\Users\\Pete\\Desktop\\Projects\\Taiwan_Receipt_App\\data\\1_digits_data.csv',
                         delimiter=",")
    labels = np.genfromtxt('C:\\Users\\Pete\\Desktop\\Projects\\Taiwan_Receipt_App\\data\\1_digits_labels.csv',
                           delimiter=",")

    training_data_all = np.reshape(data, (128, 128, x))
    training_labels_all = [int(i) for i in labels]

    label_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for i in training_labels_all:
        label_dict[i] += 1

    training_data = training_data_all[:120]
    training_labels = np.array(training_labels_all[:120])

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(248, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(training_data, training_labels, epochs=100)

    print(label_dict)
    print(len(data))
    print(training_data_all.shape)
    print(training_labels_all)

    p = -1

    test_data = training_data_all[p]
    test_label = training_labels_all[p]

    model_prediction = model.predict([[test_data]])
    prediction = np.argmax(model_prediction)

    print("The prediction of the new digit is: {}".format(prediction))
    print(int(test_label))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(test_data)

    plt.subplot(1, 2, 2)
    plt.xticks([i for i in range(10)])
    plt.bar(range(10), model_prediction[0][0:10], color="#777777")
    plt.ylim(0.0, 1.0)

    plt.show()

    return model


def web_cam():
    video_feed = cv2.VideoCapture(0)

    while True:
        ret, frame = video_feed.read()
        cv2.rectangle(frame, (60, 160), (572, 288), (0, 0, 255), 1)
        for i in range(8):
            cv2.line(frame, (60 + 64 * i, 160), (60 + 64 * i, 288), (0, 0, 255), 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Binary Camera Feed', gray)

        k = cv2.waitKey(1) & 0xFF

        if k == ord(' '):

            img_name = "open.cv_frame_{}.jpg".format(int(time.time()))
            ret, frame = video_feed.read()
            cv2.imwrite(img_name, frame)

            print("{} written!".format(img_name))

            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('image', img)
            cv2.waitKey(0)

            numbers = img[160: 288, 60: 572]
            cv2.imshow('numbers', numbers)
            cv2.waitKey(0)
            rows, cols = numbers.shape

            num1 = numbers[0:int(rows), 0:int(cols / 8)]
            num2 = numbers[0:int(rows), int(cols / 8):int(cols / 8 * 2)]
            num3 = numbers[0:int(rows), int(cols / 8 * 2):int(cols / 8 * 3)]
            num4 = numbers[0:int(rows), int(cols / 8 * 3):int(cols / 8 * 4)]
            num5 = numbers[0:int(rows), int(cols / 8 * 4):int(cols / 8 * 5)]
            num6 = numbers[0:int(rows), int(cols / 8 * 5):int(cols / 8 * 6)]
            num7 = numbers[0:int(rows), int(cols / 8 * 6):int(cols / 8 * 7)]
            num8 = numbers[0:int(rows), int(cols / 8 * 7):int(cols)]

            list_of_nums = [num1, num2, num3, num4, num5, num6, num7, num8]

            video_feed.release()
            break

    cv2.destroyAllWindows()

    return list_of_nums


def main():
    model = tf_model()

    exit_msg = ""
    while exit_msg is not "y":

        data_for_prediction = web_cam()
        prediction = []
        for digit in data_for_prediction:
            model_prediction = model.predict([[digit]])
            prediction.append(np.argmax(model_prediction))

            """Displays individual evaluations."""
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(digit)

            plt.subplot(1, 2, 2)
            plt.xticks([i for i in range(10)])
            plt.bar(range(10), model_prediction[0][0:10], color="#777777")
            plt.ylim(0.0, 1.0)

            plt.show()

        print("The prediction of the receipt is: {}".format(prediction))
        if prediction in WINNING_NUMS:
            print("Winner!")
        exit_msg = input("Do you want to exit ('y' for yes)")


if __name__ == '__main__':
    main()
