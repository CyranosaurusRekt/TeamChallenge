import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, AveragePooling3D

#  --------------------------------------------------------  #
# |                      Parameters                        | #
#  --------------------------------------------------------  #

# Part of data that will be used for training
trainPerc = .6

#  --------------------------------------------------------  #
# |                      Functions                         | #
#  --------------------------------------------------------  #

def execute(imgListTP, imgListFP):
    # Permutation of 1, 2, 3, ..., len(candidates_TP)
    r = np.random.permutation(range(1, len(imgListTP)))
    lastTrainPos = int(round(len(r) * trainPerc))
    candidates_TP_train = imgListTP[r[1:lastTrainPos]]
    candidates_TP_test = imgListTP[r[(lastTrainPos + 1):]]

    # Permutation of 1, 2, 3, ..., len(candidates_FP)
    r = np.random.permutation(range(1, len(imgListFP)))
    lastTrainPos = int(round(len(r) * trainPerc))
    candidates_FP_train = imgListFP[r[1:lastTrainPos]]
    candidates_FP_test = imgListFP[r[(lastTrainPos + 1):]]

    x_train = np.concatenate((candidates_TP_train, candidates_FP_train))
    x_test = np.concatenate((candidates_TP_test, candidates_FP_test))

    x_train = reshapeForModelInput(x_train)
    x_test = reshapeForModelInput(x_test)
    print(x_train.shape)

    # print(candidates_TP_test.shape)
    # print(candidates_FP_test.shape)
    # print(x_train.shape)

    y_train = np.concatenate((np.ones(len(candidates_TP_train)), np.zeros(len(candidates_FP_train))))
    y_test = np.concatenate((np.ones(len(candidates_TP_test)), np.zeros(len(candidates_FP_test))))

    # plt.figure(0)
    # plt.imshow(x_train[10, rAroundCentroid, :, :, 0], cmap='gray')
    # plt.show()
    #
    # for l in range(0,x_train.shape[0]):
    #     print(l[0].shape)
    #
    # for l in candidates_FP_test:
    #     plt.figure(0)
    #     plt.imshow(l[rAroundCentroid, :, :], cmap='gray')
    #     plt.show()

    # Model creation
    model = Sequential()
    model.add(Conv3D(1, (3, 3, 3), activation='relu', input_shape=(11, 11, 11, 1)))
    model.add(Conv3D(1, (3, 3, 3), activation='relu'))
    model.add(Conv3D(1, (3, 3, 3), activation='relu'))
    model.add(AveragePooling3D((2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=32,
              batch_size=32)

    score = model.evaluate(x_test, y_test, batch_size=20)

    return score


# Only reshaping for inputs with 3D samples
def reshapeForModelInput(x):
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))
    return x
