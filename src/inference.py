from model import Heart
from lmfit import minimize, Parameters


def residual(parameters, initial, time, data):
    x0, b0 = initial
    heart = Heart(x0, b0)
    tentative = heart.beat(time, parameters)
    #todo customise views --> remove non-observable signals
    residue = (tentative - data).ravel()
    return residue


class InferencedHeart(Heart):
    def inference(self, time, data):
        params = Parameters()
        params.add('eps', value=1, min=-1000., max=1000.)
        params.add('a', value=1, min=-1000., max=1000.)
        params.add('xa', value=1, min=-1000., max=1000.)
        minimized = minimize(residual, params, args=(self.initial, time, data), method='leastsq')
        names = minimized.var_names
        retval = dict()
        for name in names:
            retval[name] = minimized.params[name].value
        return retval
