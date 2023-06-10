from Walker import Walker
from utils import Timer, print_progress_bar

import numpy as np
import matplotlib as mpl
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from skimage.draw import line_aa

COLORMAPS = list(mpl.colormaps.keys())  # type: ignore


class Canvas:
    """
    The Canvas class is used to create a canvas on which a list of walkers can be added and their paths rendered.

    Parameters
    ----------
        size : numpy.ndarray
            The size of the canvas as a 1D numpy array of shape (2,).
        wrap : bool
            A boolean indicating whether the canvas should wrap around its boundaries.


    Attributes
    ----------
        canvas : numpy.ndarray
            A 3D numpy array of shape (size * 3, 3) representing the canvas.
        bounds : numpy.ndarray
            A 2D numpy array of shape (2, 2) representing the bounds of the canvas.

    Methods
    -------
        isBounded(position: np.ndarray) -> bool:
            Checks if the input position falls within the specified bounds.

        addWalkers(walkers: list[Walker]) -> np.ndarray:
            Adds a list of walkers to the canvas and renders their paths.

        compactCanvas() -> np.ndarray:
            Returns a compact version of the canvas by either cropping or summing the nine sections of the wrapped canvas.
    """

    def __init__(
        self,
        canvas_parameters: dict,
    ) -> None:
        self.size = np.asarray(canvas_parameters["size"], dtype=np.uint)
        self.wrap = bool(canvas_parameters["wrapping"])
        self.save_path = canvas_parameters["save path"]
        self.save_name = canvas_parameters["save name"]
        self.canvas = np.zeros((*self.size * 3, 3))
        self.bounds = np.array([[*self.size], [*self.size * 2]])

    def isBounded(self, position: np.ndarray) -> bool:
        """
        Checks if the input position falls within the specified bounds.

        Parameters:
            pos (numpy.ndarray): the position as a 1D numpy array of shape (2,)
            x_max (float): the maximum x value of the bounds
            y_max (float): the maximum y value of the bounds

        Returns:
            True if the position falls within the bounds, False otherwise.
        """
        return np.all(position >= self.bounds[0]) and np.all(position <= self.bounds[1])  # type: ignore

    def addWalkers(self, walkers: list[Walker]) -> np.ndarray:
        timer = Timer(len(walkers))
        print_progress_bar(
            0, timer.total, prefix="Progress:", suffix="Complete\tETA: N\\A", length=50
        )

        for i, walker in enumerate(walkers):
            print_progress_bar(
                i + 1,
                timer.total,
                prefix="Progress:",
                suffix="Complete\tETA: {}".format(timer.remains(1 + i)),
                length=50,
            )

            for j in range(len(walker.path[:-1])):
                c_pos = walker.path[j]
                n_pos = walker.path[j + 1]

                # Draw line
                rr, cc, val = line_aa(*c_pos, *n_pos)
                self.canvas[rr, cc, 0] += val * walker.height
                self.canvas[rr, cc, 1] += val * walker.height
                self.canvas[rr, cc, 2] += val * walker.height

                # Check if bounded
                if not self.isBounded(n_pos):
                    if not self.wrap:
                        break

                    if n_pos[0] < self.bounds[0, 0]:
                        walker.path[j + 1 :, 0] += self.size[0]
                    elif n_pos[0] > self.bounds[1, 0]:
                        walker.path[j + 1 :, 0] -= self.size[0]

                    if n_pos[1] < self.bounds[0, 1]:
                        walker.path[j + 1 :, 1] += self.size[1]
                    elif n_pos[1] > self.bounds[1, 1]:
                        walker.path[j + 1 :, 1] -= self.size[1]

        return self.canvas

    def foldCanvas(self) -> np.ndarray:
        canvas = np.zeros((*self.size, 3))

        # Sum all 9 sections of canvas
        if self.wrap:
            for x_sec in range(3):
                for y_sec in range(3):
                    x_i, y_i = np.array([x_sec, y_sec]) * self.size
                    x_f, y_f = np.array([x_i, y_i]) + self.size
                    canvas += self.canvas[x_i:x_f, y_i:y_f]

        # Crop canvas
        else:
            canvas = self.canvas[
                self.size[0] : self.size[0] * 2, self.size[1] : self.size[1] * 2
            ]

        self.canvas = canvas

        # Return new canvas
        return self.canvas

    def cutCanvas(self, height: int | float = 3) -> np.ndarray:
        limit = np.average(self.canvas) + height * np.std(self.canvas)
        self.canvas[np.where(self.canvas > limit)] = limit

        return self.canvas

    def scaleCanvas(self, scale: int | float = 255) -> np.ndarray:
        # Shift minimum to zero
        self.canvas -= np.min(self.canvas)

        # Scale to desired range
        self.canvas = self.canvas / np.max(self.canvas) * scale

        # Propagate scale type to data
        if type(scale) == int:
            self.canvas = self.canvas.astype(np.uint8)

        return self.canvas

    def colorCanvas(self, colormap: list[list] | str) -> np.ndarray:
        # Scale canvas to [0.0, 1.0] range
        self.scaleCanvas(scale=1.0)

        # Get desired colormap and load into pallette
        pallette = lambda pos: pos
        # Pallette from matplotlib
        if type(colormap) == str and colormap in COLORMAPS:
            cmap = mpl.cm.get_cmap(colormap)  # type: ignore

            pallette = lambda pos: cmap(pos)
            self.canvas = pallette(self.canvas[:, :, 0])[:, :, :3]

        # Pallette from http://dev.thi.ng/gradients/
        elif type(colormap) == list:
            a, b, c, d = colormap

            pallette = lambda pos: a + b * np.cos(2 * np.pi * (c * pos + d))
            self.canvas = pallette(self.canvas)

        return self.canvas

    def showCanvas(self) -> None:
        # Scale canvas to [0, 255] range
        self.scaleCanvas(scale=1.0)

        image = Image.fromarray(np.moveaxis(self.canvas, 1, 0), "RGB")
        image.show()

        return None

    def saveCanvas(self, save_info: dict = {}) -> None:
        # Scale canvas to [0, 255] range
        self.scaleCanvas(scale=255)

        saveImg = Image.fromarray(np.moveaxis(self.canvas, 1, 0), "RGB")
        self.save_name = "NewPath.png" if self.save_name == "" else self.save_name

        metadata = PngInfo()
        for key, val in save_info.items():
            metadata.add_text(str(key), str(val))

        saveImg.save(self.save_path + self.save_name, pnginfo=metadata)

        return None
