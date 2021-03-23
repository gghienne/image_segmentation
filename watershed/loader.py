from skimage import color
import matplotlib.pyplot as plt
from skimage.transform import resize


def load_images(image, size=None, show=False):
    original = (
        color.rgb2gray(
            plt.imread("./data/" + image + "/" + image + "_original.png")
        )
        * 255
    )
    truth = (
        color.rgba2rgb(
            plt.imread("./data/" + image + "/" + image + "_groundtruth.png")
        )
        * 255
    )  # .astype(int)
    seed = (
        color.rgba2rgb(
            plt.imread("./data/" + image + "/" + image + "_seed.png")
        )
        * 255
    )  # .astype(int)
    if size is not None:
        original = resize(original, size, anti_aliasing=True)
        truth = resize(truth, size, anti_aliasing=True)
        seed = resize(seed, size, anti_aliasing=True)
    original = original.astype(int)
    truth = truth.astype(int)
    seed = seed.astype(int)
    if show:
        plt.figure(figsize=(15, 15))
        plt.subplot(131), plt.imshow(original, cmap=plt.cm.gray), plt.axis(
            "off"
        ), plt.title("Original image")
        plt.subplot(133), plt.imshow(seed), plt.axis("off"), plt.title(
            "Seed image"
        )
        plt.subplot(132), plt.imshow(truth), plt.axis("off"), plt.title(
            "Groundtruth image"
        )
        plt.show()
    return original, seed, truth
