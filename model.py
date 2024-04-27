from scipy.integrate import odeint
from moods import Mood


def model(variables, t, parameters):
    try:
        eps = parameters['eps'].value
        a = parameters['a'].value
        xa = parameters['xa'].value
    except:
        eps = parameters['eps']
        a = parameters['a']
        xa = parameters['xa']
    x, b = variables
    dxdt = -1 * (x ** 3 - a * x + b) / eps
    dbdt = x - xa
    return [dxdt, dbdt]


class Heart:
    def __init__(self, x0, b0, mode='normal'):
        mood = Mood()
        self.paras = mood.getparameters('normal')
        self.x0 = x0
        self.b0 = b0

    def beat(self, t, parameters=None):
        parameters = self.paras if parameters is None else parameters
        results = odeint(model, [self.x0, self.b0], t, args=(parameters,))
        return results
