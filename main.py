#!/usr/bin/python3
import cv2
import argparse
import os
import math


DATABASE_PATH = "./database"


def euclideanDistance(x, y):
    ''' Computes the distance between n-dimensional.
    '''
    return math.sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))


def jaccardIndex(boxA, boxB):
    ''' Jaccard Index. Intersection over Union (IoU).
    Uses images bounding boxes.
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def chooseImages(imageList, number):
    ''' Uses the Jaccard distance to further refine matching results.
    '''
    outList = []
    for image in imageList:
        # While list is smaller than number
        if len(outList) < number:
            # If 1 - jaccardIndex < 0.15 appends image
            if image[2] < 0.15:
                outList.append(image)

    return outList


def readDatabase(database_path):
    # Reads the database
    databaseFileList = []
    for f in os.listdir(DATABASE_PATH):
        if f.endswith(".png"):
            databaseFileList.append([DATABASE_PATH + '/' + f, float('inf'), float('inf')])

    return databaseFileList


def computeDistances(imageList, inputImage):
    ''' Compute the distance of an input image against an input database.
    Calculates the Hu Moments euclidean distance and the Jaccard distance.
    '''
    # Gets image contour
    _, contourInput, _ = cv2.findContours(inputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Gets contour Hu Moment
    input_hu = cv2.HuMoments(cv2.moments(contourInput[1])).flatten()

    databaseFileList = imageList
    for i, f in enumerate(databaseFileList):
        # Reads image
        img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
        # Gets its contour
        _, contour, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Gets image Hu Moment
        humom = cv2.HuMoments(cv2.moments(contour[1])).flatten()
        # Euclidean distance of Hu Moment of the input image and candidate match
        # Discards the skew moment
        f[1] = euclideanDistance(humom[:7], input_hu[:7])
        # Jaccard distance
        jacc_index = jaccardIndex(cv2.boundingRect(inputImage), cv2.boundingRect(img))
        f[2] = euclideanDistance([jacc_index], [1])

    return databaseFileList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Image", type=str, help="Image to search used to search other similar images.")
    parser.add_argument("Number", type=int, help="Number of similar images to search.")
    args = parser.parse_args()

    image = args.Image
    number_images = args.Number

    # Read the database
    databaseFileList = readDatabase(DATABASE_PATH)

    # Sanity check because the number of requested files cannot be larger than the database
    if number_images >= len(databaseFileList):
        number_images = len(databaseFileList)

    # Reads input image
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Compute distances of the database images to the input image
    databaseFileList = computeDistances(databaseFileList, image)

    # Sorts the list based on the distance to the input image
    databaseFileList.sort(key=lambda x: x[1])

    # Further refine search results using jaccard index
    chosenImages = chooseImages(databaseFileList, number_images)

    # Plots input image
    cv2.imshow("input", image)

    # Show similar images
    for i in chosenImages:
        img = cv2.imread(i[0], cv2.IMREAD_GRAYSCALE)
        cv2.imshow(i[0] + ' ' + str(i[1]), img)

    cv2.waitKey()
