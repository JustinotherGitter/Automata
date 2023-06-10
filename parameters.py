canvas = {
    "size": [1920, 1080],
    "wrapping": False,
    "cut off": 0.5,
    "color map": "cividis",
    # "viridis",
    # [[0.580, 0.000, 0.220],
    #     [-0.430, 1.000, 0.200],
    #     [0.550, -0.220, 0.580],
    #     [0.920, 0.250, 0.827],],
    "save path": "C:/Users/User/Pictures/",
    "save name": "",
}

walker = {
    "walker count": 1_000,
    "location": [1920 * 1.5, 1080 * 1.5],
    "minimum angle": [60],
    "vision": [2],
    "distance": [5, 10, 20],
    "age": [100, 101],
    "height": [25, 100],
}

sierpinski = {
    "iterations": 10_000,
    "corners": "",
}

sandpile = {
    "shape": None,
    "height": 2148,
    "processor": "optimized",  # "optimized" | "animation"
    "max contents": None,
    "max history": None,
}

# "std_dev": [0.0, 0.0],
# "height_lim": -4,
# "max_height": -1,
# "smoothing": 0.1,
