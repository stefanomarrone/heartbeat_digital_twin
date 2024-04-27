from model import Heart
from inference import InferencedHeart
import numpy as np


def inferencing(initialconditions, time):
    iheart = InferencedHeart(initialconditions)


def modelling(initialconditions, time):
    x0, b0 = initialconditions
    heart = Heart(x0, b0)
    data = heart.beat(time)
    return data


if __name__ == '__main__':
    initial = (0,0)
    t = np.linspace(0, 3600, 3600*10)
    results = modelling(initial, t)


