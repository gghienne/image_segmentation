from skimage import color
import matplotlib.pyplot as plt


def load_images(image):
    original = (
        color.rgb2gray(
            plt.imread(
                "segmentation/data/" + image + "/" + image + "_original.png"
            )
        )
        * 255
    )
    truth = (
        color.rgba2rgb(
            plt.imread(
                "segmentation/data/" + image + "/" + image + "_groundtruth.png"
            )
        )
        * 255
    ).astype(int)
    print(truth.dtype)
    seed = (
        color.rgba2rgb(
            plt.imread(
                "segmentation/data/" + image + "/" + image + "_seed.png"
            )
        )
        * 255
    ).astype(int)
    return original, seed, truth
