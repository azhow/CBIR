#!/usr/bin/python3
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Image", type=str, help="Image to search used to search other similar images.")
    parser.add_argument("Number", type=int, help="Number of similar images to search.")
    args = parser.parse_args()

    image = args.Image
    number_images = args.Number

    print(image)
    print(number_images)
