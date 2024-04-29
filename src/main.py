from model import Heart, NoisyHeart
from inference import InferencedHeart
from moods import Mood
import numpy as np
from fspns import FspnGenerator

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
    # preparation
    seconds = 3600
    x0 = 1
    b0 = 0
    template = 'heartbeat_template.fspn'
    # running initial model
    initial = (x0, b0)
    t, step = np.linspace(0, seconds, seconds * 10, retstep=True)
    modelled = modelling(initial, t, 'normal')
    # inferencing the model from data
    inferenced = inferencing(initial, t, modelled)
    mood = Mood()
    results = mood.getmostlikelymood(inferenced)
    # executing FSPN model
    makoparameters = dict(inferenced)
    makoparameters['seconds'] = seconds
    makoparameters['x0'] = x0 + 500
    makoparameters['b0'] = b0 + 500
    makoparameters['b_plot'] = "b_plot.out"
    makoparameters['x_plot'] = "x_plot.out"
    makoparameters['concretefile'] = 'heartbeat_ghost.fspn'
    makoparameters['steps'] = step
    modelgenerator = FspnGenerator(template)
    modelgenerator.execute(makoparameters)



