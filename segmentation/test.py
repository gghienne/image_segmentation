from loader import load_images
from watershed import Watershed

im = "horse"  # choose between [horse,baby,soccer]

# load images
original, seed, truth = load_images(im)
# define class
w = Watershed(original, seuil=15, expand_rate=3)
# segment
w.segmentation()
w.find_fg(seed)
w.get_performances(truth)
# plot
w.plot_result(compare=seed)