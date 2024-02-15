# import the necessary packages
import numpy as np
import cv2


class ColorDescriptor:
    def __init__(self, bins):
        # Number of stored 3D histograms
        self.bins = bins

    def describe(self, image):
        # Convert the image to HSV color space and initialize
        # Features for quantifying images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # Get the size and calculate the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # Divide the image into four rectangles / segments (top left, top right, bottom right, bottom left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        # Construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # Construct a mask for each corner of the image, subtracting the center of the ellipse from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
            # Extract the color histogram from the image, and then update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # Extract the color histogram from the elliptical region and update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # Return eigenvector
        return features

    def histogram(self, image, mask):
        # The 3D color histogram is extracted from the mask area of the image using the bin number of each channel provided
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256]) # or [0, 256, 0, 256, 0, 256]
        hist = cv2.normalize(hist, hist).flatten()
        #Return histogram
        return hist


def run():
    color = ColorDescriptor((8, 8, 8))
    img = cv2.imread('G:\\Data\\pic_time\\Photos\\Mila_initial_photos\\photos\\6932617963.jpg')
    color_features = color.describe(image=img)
    print("Color features", color_features)
    print('Len of color features', len(color_features))
    # We could fist change the mask into None
    # We could manipulate the number of bins so instead of 8 colors for each chaneel we could make it 32, or between 8 to 64 bins or simply put [256] as histSize
    # change the hist range to [0,256]
    # read here https://pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
    # if we have noise in the image such as lighiting in the environment we can avoid this by using another color space such as HSV or L*a*b


if __name__ == '__main__':
    run()
