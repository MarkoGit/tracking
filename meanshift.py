import os
import cv2 as cv
import numpy as np
import argparse
import glob


def getRect(target, windowSize):
    return (int(np.ceil(target[0])) - windowSize[0] // 2,
            int(np.ceil(target[1])) - windowSize[1] // 2,
            windowSize[0],
            windowSize[1])

def getHist(image):
    mask = cv.inRange(image, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    return cv.calcHist([image], [0], mask, [180], [0, 180])

def imcrop(img, rect):
   x, y, height, width = rect
   return img[y: y + height, x: x + width]

def meanShift(image, window):
    threshold = 1
    centroid = np.asarray([0, 0])

    iterations = 100
    for iteration in range(iterations):
        updatedCentroid = np.asarray([0, 0])
        template = imcrop(image, getRect(centroid + window[:2], window[2:4]))

        indexes = np.argwhere(template > 0)
        sum = np.sum(indexes, axis = 0)
        updatedCentroid[1] = sum[0]
        updatedCentroid[0] = sum[1]
        updatedCentroid = updatedCentroid / indexes.shape[0]

        if np.linalg.norm(centroid - updatedCentroid) < threshold:
            break
        else:
            centroid = updatedCentroid

    return getRect(centroid + window[:2], window[2:4])


def sequence(path, target, windowSize):
    imagesList = sorted(glob.glob(os.path.join(path, '*.jpg')))
    imagesList = [cv.imread(image)
              for image in imagesList]

    previousImage = imagesList.pop()
    trackWindow = getRect(target, windowSize)
    trackWindow1 = trackWindow
    trackWindow2 = trackWindow

    template = imcrop(previousImage, trackWindow)
    templateHSV = cv.cvtColor(template, cv.COLOR_BGR2HSV)

    templateHist = np.array(getHist(templateHSV))

    cv.normalize(templateHist, templateHist, 0, 255, cv.NORM_MINMAX)
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    for idx, image in enumerate(imagesList):
        frame = image
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], templateHist, [0, 180], 1)

        trackWindow = meanShift(dst, trackWindow)
        _, trackWindow1 = cv.meanShift(dst, trackWindow1, term_crit)
        _, trackWindow2 = cv.CamShift(dst, trackWindow1, term_crit)

        x, y, w, h = trackWindow
        img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        x, y, w, h = trackWindow1
        img2 = cv.rectangle(img2, (x, y), (x + w, y + h), 100, 2)

        x, y, w, h = trackWindow2
        img2 = cv.rectangle(img2, (x, y), (x + w, y + h), 20, 2)

        cv.imshow('img2', img2)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k) + ".jpg", img2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', required=True)
    parser.add_argument('-target', nargs='+', type=int, required=True)
    parser.add_argument('-window', nargs='+', type=int, required=True)
    parsed = parser.parse_args()

    sequence(parsed.path, parsed.target, parsed.window)

if __name__== "__main__":
    main()
