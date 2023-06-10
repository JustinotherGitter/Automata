from utils import cartesian

import numpy as np


class Walker:
    """
    Walker class for "walkers" that determine drawing parameters on the canvas.
    A walker "steps" in a direction from its current location to its next location based on its vision.

    Parameters
    ----------
        location : np.array[int, int] | [int, int]
            The position of a walker in 2D space
        min_angle: int
            The minimum, smallest, angle the walker can turn
        vision : int
            The amount of "min_angle"'s the walker may choose from when picking a new direction
        distance : int
            The travel distance the walker steps from its current "location" to its next location
        age : int
            The iterations the walker has left before death

    Attributes
    ----------
        loc_prev : np.array[int, int] | [int, int]
            The previous location of the walker in 2D space
        theta : float
            The direction, theta, the walker is facing from the x^+ axis, anti-clockwise
        height : int
            The value left behind by the walker as it moves

    Returns
    -------
    None
    """

    def __init__(
        self,
        walker_parameters: dict,
    ) -> None:
        self.path = np.asarray([walker_parameters["location"]], dtype=np.int16)
        self.min_angle = np.random.choice(walker_parameters["minimum angle"])
        self.vis = walker_parameters["vision"]
        self.r = np.random.choice(walker_parameters["distance"])
        self.theta = np.random.randint(0, int(360 / self.min_angle)) * self.min_angle
        self.age = np.random.randint(*walker_parameters["age"])
        self.height = np.random.randint(*walker_parameters["height"])

        return None

    def walk(self, steps: int = -1) -> np.ndarray:
        if self.age <= 0:
            raise NotImplementedError(
                f"Raising an error as walk should never be called with age <= 0."
            )

        # Walk steps amount or until age == 0.
        if steps > 0:
            steps = min(steps, self.age)

        # Take one step.
        elif steps == 0:
            steps = 1

        # Walk until age == 0.
        else:
            steps = self.age

        new_path = np.zeros((steps, 2), dtype=np.int16)
        self.look()
        new_path[0] = self.path[-1] + cartesian(self.r, self.theta)
        self.age -= 1
        for step in range(1, steps):
            self.look()
            new_path[step] = np.asarray(
                new_path[step - 1] + cartesian(self.r, self.theta), dtype=np.int16
            )
            self.age -= 1

        self.path = np.append(self.path, new_path, axis=0)

        return new_path

    def look(self) -> None:
        vis: int = 0
        if type(self.vis) == list and len(self.vis) == 2:
            pass
        else:
            vis = self.vis[0] if type(self.vis) == list else self.vis  # type: ignore

        if vis > 0:
            self.theta = np.random.choice(
                [self.theta + self.min_angle * i for i in range(-vis, vis + 1)]
            )
        else:
            self.theta = (
                np.random.randint(0, int(360 / self.min_angle)) * self.min_angle
            )

        return None
