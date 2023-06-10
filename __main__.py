#!usr/bin/env python3
from Canvas import Canvas
from Walker import Walker
from Sierpinski import Sierpinski
from Sandpiles import Sandpile
import parameters as params

# import numpy as np
import logging
import sys

# Logging setup
logging.basicConfig(level=logging.INFO)


def main(mode: str):
    if mode == "walkers":
        walkers = []
        for _ in range(params.walker["walker count"]):
            # Any dependent walker params redefined here
            # walker_params['location'] = np.random.normal(size / 2, std_dev) + size

            walkers.append(Walker(params.walker))
            walkers[-1].walk()

        canvas = Canvas(params.canvas)
        canvas.addWalkers(walkers)
        canvas.foldCanvas()
        canvas.cutCanvas(params.canvas["cut off"])
        canvas.colorCanvas(colormap=params.canvas["color map"])
        # canvas.showCanvas()

        regen = params.canvas
        regen.update(params.walker)
        canvas.saveCanvas(save_info=regen)

    if mode == "sierpinski":
        sierpinski = Sierpinski(params.sierpinski)
        sierpinski.iterate()
        sierpinski.show()

    if mode == "sandpile":
        pile = Sandpile(params.sandpile)
        pile.animate() if params.sandpile["processor"] == "animation" else pile.show()


if __name__ == "__main__":
    main(sys.argv[1].lower())
