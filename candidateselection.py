import numpy as np
import SimpleITK as sitk # pip install SimpleITK
import matplotlib.pyplot as plt

#  --------------------------------------------------------  #
# |                      Parameters                        | #
#  --------------------------------------------------------  #

# Candidates cannot have an area greater than this
maxCMBSize = 10

# Maximum distance to be counted as true positive
maxDist = 6

#  --------------------------------------------------------  #
# |                      Functions                         | #
#  --------------------------------------------------------  #

# Main function to be called by user
def execute(pathPrefix):
    t2Image = sitk.ReadImage(pathPrefix + "/T2W_FFE.nii.gz")
    p1Image = sitk.ReadImage(pathPrefix + "/mri/p1T1.nii")
    p2Image = sitk.ReadImage(pathPrefix + "/mri/p2T1.nii")

    t2 = sitk.GetArrayFromImage(t2Image)
    # print(t2.shape)
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

    return [imgListTP, imgListFP]

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
