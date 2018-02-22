import numpy as np
import SimpleITK as sitk # pip install SimpleITK
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, AveragePooling3D

#  --------------------------------------------------------  #
# |                      Parameters                        | #
#  --------------------------------------------------------  #

# Candidates cannot have an area greater than this
maxCMBSize = 10

# Maximum distance to be counted as true positive
maxDist = 6

# Part of data that will be used for training
trainPerc = .6

#  --------------------------------------------------------  #
# |                      Functions                         | #
#  --------------------------------------------------------  #


# Takes 3 images and double value as input, return logical array with size of images
def getCandidates(t2, p1, p2, perc):
    # Compute the brain mask by thresholding p0 > 1.5 (arbitrary upper limit of 5)
    p1Image = sitk.GetImageFromArray(p1)
    p2Image = sitk.GetImageFromArray(p2)
    brainMaskImage = sitk.BinaryThreshold(p1Image + p2Image, 0.01, 50.0, 1, 0)

    # Get numpy arrays from the images
    brainMask = sitk.GetArrayFromImage(brainMaskImage)

    # Extract candidates
    # candidates = np.logical_and(t2 >= 0, t2 <= 1)
    allValues = t2
    allValues = allValues.flatten()
    allValues = np.sort(allValues)
    thresh = allValues[round(allValues.size * perc) - 1]
    candidatesArray = np.logical_and(np.logical_and(t2 >= 0, t2 <= thresh), brainMask) * 1

    return candidatesArray


# Get coordinates of annotations in form of [Z Y X]
def getAnnotations(pathPrefix):
    # Get coordinates
    coord = np.loadtxt(pathPrefix+"/T2W_FFE.xmark", dtype="int")

    # Z Y X
    coord = np.array([coord[:, 2], coord[:, 1], coord[:, 0]]).T

    return coord


# Display all CMBs given coordinates, image and
# radius to display around CMBs
def showCMBs(npImage, coord, r):
    for i in range(0, coord.shape[0]):
        plt.figure(i)
        c = coord[i, :]  # Getting the ith CMB
        plt.imshow(npImage[c[0], c[1]-r:c[1]+r, c[2]-r:c[2]+r], cmap="gray")
        plt.show()

# Get connected component list from binary array
# Returns a list of n entries where n is the number
# of connected components. Each index contains the
# centroid of the image.
def getConnected(candidatesArray):
    candidatesImg = sitk.GetImageFromArray(candidatesArray)
    candidatesCCObj = sitk.ConnectedComponent(candidatesImg, True)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(candidatesCCObj, candidatesImg)

    candidatesCC = []
    for l in stats.GetLabels():
        sz = stats.GetPhysicalSize(l)
        if sz < maxCMBSize:
            x = np.array(stats.GetCentroid(l))
            x = x.astype(int)
            x = np.array([x[2], x[0], x[1]])
            candidatesCC.append(x)
        # else:
        #     print("too large " + str(sz))

    return candidatesCC


# Get a list of 3D Images around centroids
def getListOf3DImagesAroundCentroids(npimage, candidatesCC, r):
    list3DImages = []

    sz = npimage.shape
    for c in candidatesCC:
        z = c[0]
        x = c[1]
        y = c[2]

        img = npimage[max(z - r, 0):min(sz[0], z + r) + 1
                          , max(y - r, 0):min(sz[1], y + r) + 1
                          , max(x - r, 0):min(sz[2], x + r) + 1]

        # Fill img with zeros at borders if img.shape[] != r x r x r
        if z < r:
            zrs = np.zeros(((r-z), img.shape[1], img.shape[2]), dtype=int)
            img = np.concatenate((zrs, img))
        if img.shape[0] < 2*r+1:
            zrs = np.zeros((2*r+1-img.shape[0], img.shape[1], img.shape[2]))
            img = np.concatenate((img, zrs))

        if x < r:
            zrs = np.zeros((img.shape[0], (r - x), img.shape[2]))
            img = np.concatenate((zrs, img))
        if img.shape[1] < 2 * r + 1:
            zrs = np.zeros((img.shape[0], (2 * r + 1 - img.shape[1]), img.shape[2]))
            img = np.concatenate((img, zrs))

        if y < r:
            zrs = np.zeros((img.shape[0], img.shape[1], (r - y)))
            img = np.concatenate((zrs, img))
        if img.shape[2] < 2 * r + 1:
            zrs = np.zeros((img.shape[0], img.shape[1], (2 * r + 1 - img.shape[2])))
            img = np.concatenate((img, zrs))

        list3DImages.append(img)

    return np.array(list3DImages)


# Get either true or false positives. Set bool == True for TP, bool == False for FP
def getXP(candidates, annotations, bool):
    ret = []

    print(str(bool) + " Positives:")
    annotationFreq = np.zeros(len(annotations))

    for c in candidates:
        found = False

        annCnt = -1;
        for a in annotations:
            annCnt += 1

            dist = np.sqrt(np.sum(np.power(c - a, 2)))
            if dist < maxDist:
                # print(c)
                # print(a)
                # print(' ')
                if bool:
                    annotationFreq[annCnt] = annotationFreq[annCnt] + 1

                    ret.append(c)
                    found = True
                    break
        if not bool and not found:
            ret.append(c)

    print(annotationFreq), print(' ')
    return ret


# Get true positive candidates
def getTP(candidates, annotations):
    return getXP(candidates, annotations, True)


# Get true positive candidates
def getFP(candidates, annotations):
    return getXP(candidates, annotations, False)


# Only reshaping for inputs with 3D samples
def reshapeForModelInput(x):
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))
    return x

#  --------------------------------------------------------  #
# |                     Main program                       | #
#  --------------------------------------------------------  #

# Load the images
nr = '015'
pathPrefix = "Data/"+nr
t2Image = sitk.ReadImage(pathPrefix+"/T2W_FFE.nii.gz")
p1Image = sitk.ReadImage(pathPrefix+"/mri/p1T1.nii")
p2Image = sitk.ReadImage(pathPrefix+"/mri/p2T1.nii")

t2 = sitk.GetArrayFromImage(t2Image)
p1 = sitk.GetArrayFromImage(p1Image)
p2 = sitk.GetArrayFromImage(p2Image)

annotations = getAnnotations(pathPrefix)

candidatesArray = getCandidates(t2, p1, p2, .8)

# showCMBs(t2, annotations, 25)

candidates = getConnected(candidatesArray)

# True positives (list of centroids)
candidates_TP = getTP(candidates, annotations)

# False positives (list of centroids)
candidates_FP = getFP(candidates, annotations)

rAroundCentroid = 5  # radius
imgListTP = getListOf3DImagesAroundCentroids(t2, candidates_TP, rAroundCentroid)
imgListFP = getListOf3DImagesAroundCentroids(t2, candidates_FP, rAroundCentroid)

# Permutation of 1, 2, 3, ..., len(candidates_TP)
r = np.random.permutation(range(1, len(candidates_TP)))
lastTrainPos = int(round(len(r)*trainPerc))
candidates_TP_train = imgListTP[r[1:lastTrainPos]]
candidates_TP_test = imgListTP[r[(lastTrainPos+1):]]

# Permutation of 1, 2, 3, ..., len(candidates_FP)
r = np.random.permutation(range(1, len(candidates_FP)))
lastTrainPos = int(round(len(r)*trainPerc))
candidates_FP_train = imgListFP[r[1:lastTrainPos]]
candidates_FP_test = imgListFP[r[(lastTrainPos+1):]]

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

print(score)


