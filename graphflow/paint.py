from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from math import exp, pow
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import KolmogorovSolver

# np.set_printoptions(threshold=np.inf)
graphCutAlgo = {"ap": augmentingPath,
                "pr": pushRelabel
                }
SIGMA = 30
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)
SEGCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 1
LOADSEEDS = False


# drawing = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plantSeed(image,mode):
    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Paint " + pixelType + mode
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1  # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    print("Current mode is %s"%mode)
    if mode== "Seed":
        paintSeeds(BKG)
    return seeds, image




def seedImage(image,pathname):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    seeds, seededImage = plantSeed(image,"Seed")
    np.savez(pathname + "_term", seeds)
    return graph, seededImage

def paintGTImage(image,pathname):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    seeds, seededImage = plantSeed(image,"GT")
    np.savez(pathname + "_gt", seeds)
    return graph, seededImage



def paintimg(imagefile, size=(600, 600), algo="ff",mode="GTpaint"):

    #mode="GTpaint"
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)

    if mode=="seedpaint":

        graph, seededImage = seedImage(image,pathname)
        print("graph shape:", graph.shape)
        cv2.imwrite(pathname + "seeded.jpg", seededImage)
    elif mode=="GTpaint":
        graph, seededImage = paintGTImage(image, pathname)
        print("graph shape:", graph.shape)
        cv2.imwrite(pathname + "GT.jpg", seededImage)


def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--mode","-m",default="GTpaint")
    parser.add_argument("--size", "-s",
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="pr", type=algorithm)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    paintimg(args.imagefile, (args.size, args.size), args.algo,mode = args.mode)






