import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from skimage import filters
import random as rd
import time
from skimage.morphology import square, erosion


class Watershed:
    def __init__(self, image):
        self.image = image
        self.grad = np.abs(filters.sobel(self.image))
        self.labels = None
        self.foreground = None
        self.time = None
        self.dice = None

    def segmentation(self, seuil=0):
        self.grad = np.abs(filters.sobel(self.image))
        self._seuilling(seuil)
        t0 = time.time()
        image = self.grad
        len_x, len_y = np.shape(image)
        labels = np.ones((len_x, len_y)) * (
            -2
        )  # -2: Not visited, -1: Visited, not labeled yet
        label_counter = 0
        h_max = image.max()

        for i in trange(len_x):
            for j in range(len_y):
                if (
                    labels[i, j] == -2
                ):  # If the pixel has no label yet then compute it else pass

                    # Comppute the steepest path
                    xk = np.array([i, j])
                    hk = image[xk[0], xk[1]]
                    path = [np.array(xk)]

                    while True:
                        # Find the steepest slope
                        hkp1 = h_max + 1
                        if xk[0] > 0 and image[xk[0] - 1, xk[1]] <= hkp1:
                            xkp1 = [xk[0] - 1, xk[1]]
                            hkp1 = image[xk[0] - 1, xk[1]]
                        if xk[1] > 0 and image[xk[0], xk[1] - 1] <= hkp1:
                            xkp1 = [xk[0], xk[1] - 1]
                            hkp1 = image[xk[0], xk[1] - 1]
                        if (
                            xk[0] < len_x - 1
                            and image[xk[0] + 1, xk[1]] <= hkp1
                        ):
                            xkp1 = [xk[0] + 1, xk[1]]
                            hkp1 = image[xk[0] + 1, xk[1]]
                        if (
                            xk[1] < len_y - 1
                            and image[xk[0], xk[1] + 1] <= hkp1
                        ):
                            xkp1 = [xk[0], xk[1] + 1]
                            hkp1 = image[xk[0], xk[1] + 1]

                        # The path meet himself and is on a plateau
                        if labels[xkp1[0], xkp1[1]] == -1 and hkp1 == hk:
                            hk, xk, path, labels, status = self._flood_plateau(
                                image, labels, path, xkp1
                            )
                            if status == "complete":
                                break

                        # No downgoing slope found, stuck in a minimum.
                        # Or the path meet himself and is not on a plateau.
                        if hkp1 > hk or (
                            labels[xkp1[0], xkp1[1]] == -1 and hkp1 != hk
                        ):
                            path = np.array(path)
                            labels[path[:, 0], path[:, 1]] = label_counter
                            label_counter += 1
                            break

                        # The path meet an already explored path
                        if labels[xkp1[0], xkp1[1]] >= 0:
                            path = np.array(path)
                            labels[path[:, 0], path[:, 1]] = labels[
                                xkp1[0], xkp1[1]
                            ]
                            break

                        # The path meet an already explored path
                        if labels[xkp1[0], xkp1[1]] == -2:
                            path.append(np.array(xkp1))
                            labels[xkp1[0], xkp1[1]] = -1
                            hk = hkp1
                            xk = np.array(xkp1)
        self.time = time.time() - t0
        self.labels = labels

    def _flood_plateau(self, image, labels, path, xkp1):
        len_x, len_y = np.shape(image)
        xk = path[-1]
        hk = image[xk[0], xk[1]]
        h_max = image.max()

        to_be_visited = [np.array(xkp1), path[-1]]

        xkp1 = None
        hkp1 = h_max + 1
        label = None
        while to_be_visited != []:
            xk = to_be_visited.pop()
            if (
                xk[0] > 0
                and image[xk[0] - 1, xk[1]] <= hk
                and labels[xk[0] - 1, xk[1]] != -1
            ):
                if image[xk[0] - 1, xk[1]] == hk:
                    if labels[xk[0] - 1, xk[1]] == -2:
                        to_be_visited.append(np.array([xk[0] - 1, xk[1]]))
                        path.append(np.array([xk[0] - 1, xk[1]]))
                        labels[xk[0] - 1, xk[1]] = -1
                    else:
                        label = labels[xk[0] - 1, xk[1]]
                elif hkp1 > image[xk[0] - 1, xk[1]]:
                    hkp1 = image[xk[0] - 1, xk[1]]
                    xkp1 = np.array([xk[0] - 1, xk[1]])
            if (
                xk[1] > 0
                and image[xk[0], xk[1] - 1] <= hk
                and labels[xk[0], xk[1] - 1] != -1
            ):
                if image[xk[0], xk[1] - 1] == hk:
                    if labels[xk[0], xk[1] - 1] == -2:
                        to_be_visited.append(np.array([xk[0], xk[1] - 1]))
                        path.append(np.array([xk[0], xk[1] - 1]))
                        labels[xk[0], xk[1] - 1] = -1
                    else:
                        label = labels[xk[0], xk[1] - 1]
                elif hkp1 > image[xk[0], xk[1] - 1]:
                    hkp1 = image[xk[0], xk[1] - 1]
                    xkp1 = np.array([xk[0], xk[1] - 1])
            if (
                xk[0] < len_x - 1
                and image[xk[0] + 1, xk[1]] <= hk
                and labels[xk[0] + 1, xk[1]] != -1
            ):
                if image[xk[0] + 1, xk[1]] == hk:
                    if labels[xk[0] + 1, xk[1]] == -2:
                        to_be_visited.append(np.array([xk[0] + 1, xk[1]]))
                        path.append(np.array([xk[0] + 1, xk[1]]))
                        labels[xk[0] + 1, xk[1]] = -1
                    else:
                        label = labels[xk[0] + 1, xk[1]]
                elif hkp1 > image[xk[0] + 1, xk[1]]:
                    hkp1 = image[xk[0] + 1, xk[1]]
                    xkp1 = np.array([xk[0] + 1, xk[1]])
            if (
                xk[1] < len_y - 1
                and image[xk[0], xk[1] + 1] <= hk
                and labels[xk[0], xk[1] + 1] != -1
            ):
                if image[xk[0], xk[1] + 1] == hk:
                    if labels[xk[0], xk[1] + 1] == -2:
                        to_be_visited.append(np.array([xk[0], xk[1] + 1]))
                        path.append(np.array([xk[0], xk[1] + 1]))
                        labels[xk[0], xk[1] + 1] = -1
                    else:
                        label = labels[xk[0], xk[1] + 1]
                elif hkp1 > image[xk[0] - 1, xk[1]]:
                    hkp1 = image[xk[0], xk[1] + 1]
                    xkp1 = np.array([xk[0], xk[1] + 1])

        if label is not None:
            path = np.array(path)
            labels[path[:, 0], path[:, 1]] = label
            status = "complete"
        else:
            status = "running"

        return hkp1, xkp1, path, labels, status

    def _seuilling(self, seuil):
        u = self.grad.copy()
        m, M = u.min(), u.max()
        u = np.round((u - m) / M * 255)
        self.grad = u * (u > seuil)

    def find_fg(self, seed, expand_rate=0, fg=True):
        u = self.labels.copy()
        fg_ind = np.where(seed[:, :, 2] == 255)
        # bg_ind = np.where(seed[:, :, 1] > seed[:, :, 2])
        for i in range(len(fg_ind[0])):
            u[self.labels == self.labels[fg_ind[0][i], fg_ind[1][i]]] = -1
        u[u != -1] = 1
        self.labels = u
        self._expand_labels(expand_rate)
        if fg:
            self._fg_transform()

    def _expand_labels(self, rate):
        expand_size = int((self.image.size ** 0.5) * rate / 100) + 1
        self.labels = erosion(self.labels, square(expand_size))

    def _fg_transform(self):
        u = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=int)
        for i in range(3):
            u[:, :, i] = self.image
        fg_ind = np.where(self.labels == -1)
        for i in range(len(fg_ind[0])):
            # print(fg_ind[0][i], fg_ind[1][i])
            u[fg_ind[0][i], fg_ind[1][i], :] = np.array([0, 0, 255])
        self.foreground = u

    def _compute_dice(self, truth):
        A = truth[:, :, 0] == 255
        B = self.foreground[:, :, 2] == 255
        TP = len(np.nonzero(A * B)[0])
        FN = len(np.nonzero(A * (~B))[0])
        FP = len(np.nonzero((~A) * B)[0])
        DICE = 0
        if (FP + 2 * TP + FN) != 0:
            DICE = float(2) * TP / (FP + 2 * TP + FN)
        self.dice = DICE * 100

    def plot_result(self, compare=None):
        if compare is None:
            compare = self.image
            cm = plt.cm.gray
        else:
            cm = None
        if self.foreground is None:
            rd_color = lambda: np.array(
                [rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)]
            )
            labels = self.labels
            nb_segments = int(labels.max() + 1)
            res = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=int)
            for k in range(nb_segments):
                color = rd_color()
                res[labels == k, :] = color
        else:
            res = self.foreground
        plt.figure(figsize=(15, 15))
        plt.subplot(121)
        plt.imshow(compare, cmap=cm)
        plt.title("Original image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(res)
        plt.title("Segmentation result")
        plt.axis("off")
        plt.show()

    def get_result(self):
        if self.labels is None:
            return self.image
        elif self.foreground is None:
            rd_color = lambda: np.array(
                [rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)]
            )
            labels = self.labels
            nb_segments = int(labels.max() + 1)
            res = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=int)
            for k in range(nb_segments):
                color = rd_color()
                res[labels == k, :] = color
            return res
        return self.foreground

    def get_performances(self, truth, display=True):
        self._compute_dice(truth)
        print("Runtime: ", np.round(self.time, decimals=2), "s")
        print("Dice   : ", np.round(self.dice, decimals=2), "%")
        return self.time, self.dice