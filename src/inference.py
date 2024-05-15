import lmfit

from src.model import Heart
from lmfit import minimize, Parameters, Model
from random import uniform

def residual(parameters, initial, time, data):
    x0, b0 = initial
    heart = Heart(x0, b0)
    tentative = heart.beat(time, parameters)
    #todo customise views --> remove non-observable signals
    residue = (tentative - data).ravel()
    return residue


class InferencedHeart(Heart):
    def inference(self, time, data, methodtype):
        params = Parameters()
        params.add('eps', min=0.01, max=100, value= uniform(0.01,100))
        params.add('T', min=0.01, max=100, value = uniform(0.01,100))
        params.add('xa', min=0.01, max=100, value = uniform(0.01,100))
        retval = dict()
        errorstate = True
        counter = 0
        report = None
        while errorstate and counter < 5:
            try:
                minimised = minimize(residual, params, args=(self.initial, time, data), method=methodtype)
                errorstate = False
            except ValueError:
                counter += 1
            except RuntimeError:
                counter += 1
            else:
                report = lmfit.fit_report(minimised.params)
                names = minimised.var_names
                for name in names:
                    retval[name] = minimised.params[name].value
        return retval, errorstate, report
