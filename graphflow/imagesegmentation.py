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
SF = 10
LOADSEEDS = False
# drawing = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image,pathname):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    #seeds, seededImage = plantSeed(image)
    seeds=np.load(pathname + "_term.npz")['arr_0']
    makeTLinks(graph, seeds, K)
    return graph

def addSegment(graph,x,y,val):
    graph[x][y] = val
    graph[y][x] = val

def addTlinks(graph,seeds,i,j,nodeid,val):
    if seeds[i][j] == OBJCODE:
        # graph[x][source] = K
        graph[SOURCE][nodeid] = val
    elif seeds[i][j] == BKGCODE:
        graph[nodeid][SINK] = val

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                addSegment(graph,x,y,bp)
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                addSegment(graph,x,y,bp)
                K = max(K, bp)
    return K



def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            addTlinks(graph,seeds,i,j,x,K)
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)



def displayCut(image, solver):


    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cut_array=np.zero((r,c))
    for i in range(solver.Nvertex-2):
            if solver.Tree[i]==1:
                ix=int(i%c)
                iy=int(i/c)
                image[iy][ix]=SEGCOLOR
                cut_array[iy][ix] = 1

    return image,cut_array

def displaySegmentation(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image
    


def imageSegmentation(imagefile, size=(600, 600), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    graph = buildGraph(image,pathname)
    print("graph shape:",graph.shape)

    global SOURCE, SINK
    SOURCE += len(graph) 
    SINK   += len(graph)

    solver = KolmogorovSolver(graph, SOURCE, SINK)
    solver.Kolmogorov()
    #cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    #print("cuts:")
    #print(cuts)

    cuts=solver.cuts
    print("maxflow")
    print(solver.maxflow)
    print ("cuts:")
    print (cuts)
    image,cutarray = displayCut(image, solver)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    np.savez(pathname + "_seg", cutarray)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print ("Saved image as", savename)
    

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="pr", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
    





