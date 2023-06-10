import numpy as np
import matplotlib.pyplot as plt


class Sierpinski:
    def __init__(
        self,
        sierpinski_parameters: dict,
        # iterations: int = 10_000,
        # corners: list | np.ndarray = None,  # type: ignore
    ) -> None:
        self.iterations = sierpinski_parameters["iterations"]

        if isinstance(sierpinski_parameters["corners"], (list, np.ndarray)):
            self.corners = sierpinski_parameters["corners"]
        else:
            self.corners = [[0, 0], [100, 0], [50, 50 * np.tan(np.pi / 3)]]

        self.board = np.zeros([self.iterations, 2])

        self.board[: len(self.corners)] = self.corners
        self.board[len(self.corners)] = np.average(self.corners, axis=0)

        return None

    def iterate(self) -> None:
        for i in range(len(self.corners) + 1, len(self.board)):
            moveTo = np.random.randint(len(self.corners))
            self.board[i] = list(
                np.average([self.corners[moveTo], self.board[i - 1]], axis=0)
            )

        return None

    def show(self) -> None:
        plt.plot(self.board[:, 0], self.board[:, 1], ".", markersize=0.8)
        plt.axis("equal")
        plt.show()

        return None
