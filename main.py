import candidateselection as cs
import cnn

#  --------------------------------------------------------  #
# |                     Main program                       | #
#  --------------------------------------------------------  #

# Load the images
nr = '015'
pathPrefix = "Data/"+nr

otp = cs.execute(pathPrefix)
imgListTP = otp[0]
imgListFP = otp[1]

otp2 = cnn.execute(imgListTP, imgListFP)
print(otp2)
