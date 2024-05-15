from scipy.integrate import odeint
from src.moods import Mood
import numpy as np
import math

def model(variables, t, parameters):
    try:
        eps = parameters['eps'].value
        T = parameters['T'].value
        xa = parameters['xa'].value
    except:
        eps = parameters['eps']
        T = parameters['T']
        xa = parameters['xa']
    x, b = variables
    dxdt = -1 * (x ** 3 - T * x + b) / eps
    dbdt = x - xa
    return [dxdt, dbdt]


class Heart:
    def __init__(self, x0, b0, mode='normal'):
        mood = Mood()
        self.paras = mood.getparameters(mode)
        self.x0 = x0
        self.b0 = b0
        self.initial = (x0, b0)


    def beat(self, t, parameters=None):
        parameters = self.paras if parameters is None else parameters
        results = odeint(model, [self.x0, self.b0], t, args=(parameters,))
        return results



class NoisyHeart(Heart):
    def __init__(self, x0, b0, mode='normal', snratio=0.001):
        super(NoisyHeart, self).__init__(x0, b0, mode)
        self.snr = snratio

    def beat(self, t, parameters=None):
        signal = super(NoisyHeart, self).beat(t, parameters)
        rms = [np.mean(s**2) for s in signal.transpose()]
        std = [math.sqrt(r/self.snr) for r in rms]
        noise = [np.random.normal(0, s, size=len(t)) for s in std]
        noise = np.array(noise)
        retval = signal + noise.transpose()
        return retval
