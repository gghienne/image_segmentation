from skimage import color
from watershed import Watershed
import matplotlib.pyplot as plt

# load image, truth and seed
image = "baby"
original = color.rgb2gray(
    plt.imread("data/" + image + "/" + image + "_original.jpg")
)
truth = plt.imread("data/" + image + "/" + image + "_groundtruth.jpg")
seed = plt.imread("data/" + image + "/" + image + "_seed.jpg")

# define class
w = Watershed(original, seuil=70)

# segment
w.segmentation()
w.find_fg(seed)
w.get_performances(truth)

# plot
w.plot_result(compare=truth)