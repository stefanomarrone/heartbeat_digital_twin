from model import Heart, NoisyHeart
from inference import InferencedHeart
from moods import Mood
import numpy as np


def inferencing(initialconditions, time, realdata):
    x0, b0 = initialconditions
    iheart = InferencedHeart(x0, b0)
    results = iheart.inference(time, realdata)
    return results


def modelling(initialconditions, time, condition):
    x0, b0 = initialconditions
    heart = Heart(x0, b0, condition)
    data = heart.beat(time)
    return data


def noisymodelling(initialconditions, time, condition, snr):
    x0, b0 = initialconditions
    heart = NoisyHeart(x0, b0, condition, snr)
    data = heart.beat(time)
    return data


if __name__ == '__main__':
    initial = (0,0)
    t = np.linspace(0, 3600, 3600*10)
    results = modelling(initial, t, 'coffee')
    inferenced = inferencing(initial, t, results)
    mood = Mood()
    results = mood.getmostlikelymood(inferenced)
    print(results)


