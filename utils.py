import numba as nb
import numpy as np
import datetime


# numba optimized functions
@nb.jit(nopython=True)
def cartesian(r: float, theta: float) -> np.ndarray:
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))

    return np.array((x, y), dtype=np.int32)


@nb.jit(nopython=True)
def polar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2 * np.pi

    return np.array([r, theta], dtype=np.float32)


# Timing and progress bar helper functions
def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
    print_end: str = "        \r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


class Timer(object):
    def __init__(
        self,
        total: int = 0,
    ) -> None:
        """
        Initializes the Timer object

        :param  total       : The amount you plan to iterate over
        :type   total       : int
        """
        self.start = datetime.datetime.now()
        self.total = total

    def remains(
        self,
        done: int | float,
    ):
        """
        Calculates the remaining time for the given amount of iterations

        :param  done        : The amount of iterations have been completed
        :return:time left   : The time predicted to still remain
        :rtype  str
        """
        now = datetime.datetime.now()
        left = (self.total - done) * (now - self.start) / done
        sec = int(left.total_seconds())
        if sec < 60:
            return "{} seconds".format(sec)
        else:
            return "{} minutes".format(int(sec / 60))

    def elapsed(self):
        """
        Calculates the time passed for the amount of completed iterations

        :return:time passed : The time calculated to have passed
        :rtype  timedelta object
        """
        return datetime.datetime.now() - self.start
