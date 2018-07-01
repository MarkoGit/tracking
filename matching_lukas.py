import glob
import os

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse

def imcrop(img, rect):
   x, y, height, width = rect

   return img[y:y+ height, x:x+width]

# OpenCV
def matchTemplate(image, width, height, template, method):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(grayImage, grayTemplate, method)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    if method == cv.TM_SQDIFF:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + width, topLeft[1] + height)

    return topLeft, bottomRight

# Custom template matching
def templateMatch(image, template, method = 'ssd'):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    if method == 'ncc':
        minError = 0
    else:
        minError = sys.maxsize

    position = (0, 0)
    for y in range(grayImage.shape[0] - grayTemplate.shape[0]):
        for x in range(grayImage.shape[1] - grayTemplate.shape[1]):
            cropped = imcrop(grayImage, (x, y, grayTemplate.shape[0], grayTemplate.shape[1]))
            if method == 'ssd':
                error = ssd(cropped, grayTemplate)
            elif method == 'sad':
                error = sad(cropped, grayTemplate)
            elif method == 'ncc':
                error = ncc(cropped, grayTemplate)

            if method == 'ncc':
                if error > minError:
                    if error == 1:
                        return (x, y)
                    minError = error
                    position = (x, y)
            else:
                if error < minError:
                    if error == 0:
                        return (x, y)
                    minError = error
                    position = (x, y)
    return position


def ssd(A,B):
    squares = (A[:,] - B[:,]) ** 2
    return np.sum(squares)

def sad(A,B):
    diff = (A[:,] - B[:,])
    return np.sum(diff)

def ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def match(image, rect):
    croppedImage = imcrop(image, rect)

    _, width, height = croppedImage.shape[::-1]

    x, y = templateMatch(image, croppedImage, "ncc")

    topLeft = (x, y)
    bottomRight = (x + width, y + height)

    imageToShow = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.rectangle(imageToShow, topLeft, bottomRight, 255, 2)
    plt.subplot(121), plt.imshow(imageToShow)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv.cvtColor(croppedImage, cv.COLOR_BGR2RGB))
    plt.title('Template Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def imshow(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# Lucas Kanade
def translatedImageInv(image, p):
    # Affine for translation only
    # M = np.float32([[1, 0, p[0]],
    #                 [0, 1, p[1]]])
    M = np.float32([[1 + p[0], p[2], p[4]],
                    [p[1], 1 + p[3],  p[5]]])

    M = cv.invertAffineTransform(M)
    if len(image.shape) == 2:
        rows, cols = image.shape
    else:
        rows, cols, _ = image.shape
    translated = cv.warpAffine(image, M, (cols, rows))
    return translated

def makeJacobian(array):
    # jacobian for translation only
    # lambd = lambda x: np.array([[1, 0],
    #                            [0, 1]])
    lambd = lambda x: np.array([[x[0], 0, x[1], 0, 1, 0],
                            [0, x[0], 0, x[1], 0, 1]])

    return np.apply_along_axis(lambd, 1, array)

def lucasKanade(image, template, rect):
    params = np.zeros(6)

    imageCurrent = image.copy()
    x, y, width, height = rect
    rect = (x - 1, y - 1, width, height)

    for i in range(100):
        image = translatedImageInv(imageCurrent, params)
        gy, gx = np.gradient(image)

        gx_w = imcrop(gx, rect)
        gy_w = imcrop(gy, rect)

        candidate = imcrop(image, rect)
        errorImage = (template - candidate)
        errorImageRemapped = np.tile(errorImage.flatten('F'), (len(params), 1)).T

        X, Y = np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1]))
        coords2d = np.array([X.flatten('F')+1, Y.flatten('F')+1]).T
        jacobian = makeJacobian(coords2d)

        steepest = np.asarray([np.asarray(grad).dot(jacobian[i]) for i, grad in enumerate(zip(gx_w.flatten('F'), gy_w.flatten('F')))])
        hessian = steepest.T.dot(steepest)

        costFunction = np.sum(np.multiply(steepest, errorImageRemapped), axis = 0)

        dp = np.linalg.inv(hessian).dot(costFunction.T)
        params = params + dp.T

        if (np.linalg.norm([dp[4], dp[5]])) < 0.1:
            print("success")
            break

        # print("DP - ", np.linalg.norm([dp[4], dp[5]]))

    return params

def getRect(target, window_size):
    return (int(np.ceil(target[0])) - window_size[0] // 2,
            int(np.ceil(target[1])) - window_size[1] // 2,
            window_size[0],
            window_size[1])

def sequence(path, target, windowSize):
    imagesList = sorted(glob.glob(os.path.join(path, '*.jpg')))
    currentImage = cv.imread(imagesList.pop(0), 0)
    currentImage = np.array(currentImage, dtype='float64')

    rect = getRect(target, windowSize)

    template = imcrop(currentImage, rect= rect)

    image = cv.imread(imagesList[0], 0)
    image =  np.array(image, dtype='float64')

    for idx, img in enumerate(imagesList):
        nextImage = cv.imread(img, 0)
        nextImage = np.array(nextImage, dtype='float64')
        nextImageCopy = nextImage.copy()

        params = lucasKanade(nextImageCopy, template, rect=rect)

        x, y, width, height = rect

        affine = np.array([[1 + params[0], params[2], params[4]], [params[1], 1 + params[3], params[5]]])
        target = affine.dot(np.append(np.array([x, y]), 1))

        rect = (int(round(target[0])), int(round(target[1])), width, height)

        template = imcrop(nextImage, rect)

        topLeft = (x, y)
        bottomRight = (x + width, y + height)
        cv.rectangle(nextImageCopy, topLeft, bottomRight, 255, 2)
        imshow(nextImageCopy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', required=True)
    parser.add_argument('-target', nargs='+', type=int, required=True)
    parser.add_argument('-window', nargs='+', type=int, required=True)
    parsed = parser.parse_args()

    sequence(parsed.path, parsed.target, parsed.window)



if __name__== "__main__":
    main()