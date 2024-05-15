import time
import numpy as np
import pandas as pd
import requests
from src.fspns import FspnGenerator
from src.inference import InferencedHeart
from src.model import Heart, NoisyHeart
from src.moods import Mood
from src.reporting import computeerror, plotting_results, plotting_performance, plotting_prediction, reporting


def inferencing(initialconditions, time, realdata, methodname):
    x_initial, b_initial = initialconditions
    iheart = InferencedHeart(x_initial, b_initial)
    results, errorstate, report = iheart.inference(time, realdata, methodname)
    return results, errorstate, report


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


def ntfy(msg):
    requests.post("https://ntfy.sh/alerts-heartbeat-code", data=msg.encode(encoding='utf-8'))


def influenceanalysis():
    conditionrepo = Mood()
    conditionrepo.addmood('normal', {'eps': 0.5, 'T': 10, 'xa': 0.1})
    conditionrepo.addmood('coffee', {'eps': 0.1, 'T': 50, 'xa': 0.1})
    conditionrepo.addmood('sleeping', {'eps': 0.05, 'T': 4, 'xa': 0.1})
    conditionrepo.trainclassfier()
    return conditionrepo


def getMethods():
    #methods = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping', 'ampgo', 'nelder',
    # 'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs', 'tnc', 'trust-ncg', 'trust-exact', 'trust-krylov',
    # 'trust-constr', 'dogleg', 'slsqp', 'emcee', 'shgo', 'dual_annealing']
    methodlist = ['leastsq', 'least_squares', 'powell']
    return methodlist


if __name__ == '__main__':
    # preparation
    seconds = 10
    x0 = 1
    b0 = 0
    template = 'resources/heartbeat_template.fspn'
    ntfy("Starting")
    ntfy('Influence Analysis')
    mood = influenceanalysis()
    ntfy('Inference & Simulation')
    methods = getMethods()
    # running initial model
    initial = (x0, b0)
    t, step = np.linspace(0, seconds, seconds * 10, retstep=True)
    result_db = dict()
    for mood_name in mood.moods():
        result_db[mood_name] = dict()
        modelled = modelling(initial, t, mood_name)
        initial_dataframe = pd.DataFrame(list(modelled))
        initial_dataframe.columns = ['ode x', 'ode b']
        initial_dataframe.insert(2, "ode time", t)
        initial_dataframe.to_csv('output/' + mood_name + '.csv', index=True)
        for method in methods:
            # inferencing the model from data
            starttime = time.time()
            inferenced, errorstate, report = inferencing(initial, t, modelled, method)
            stoptime = time.time()
            single_results = dict()
            single_results['report'] = report
            single_results['inferencetime'] = stoptime - starttime
            single_results['errorstate'] = errorstate
            if not errorstate:
                mostprobable = mood.getmostlikelymood(inferenced)
                # executing FSPN model
                makoparameters = dict(inferenced)
                makoparameters['seconds'] = seconds
                makoparameters['x0'] = x0 + 500
                makoparameters['b0'] = b0 + 500
                makoparameters['b_plot'] = "output/b_plot.out"
                makoparameters['x_plot'] = "output/x_plot.out"
                makoparameters['concretefile'] = 'output/heartbeat_temp.fspn'
                makoparameters['steps'] = step
                modelgenerator = FspnGenerator(template)
                starttime = time.time()
                analysisdata = modelgenerator.execute(makoparameters)
                stoptime = time.time()
                relativeerror = computeerror(mood.get(mood_name), dict(inferenced))
                single_results['recognized'] = mostprobable
                single_results['simulationtime'] = stoptime - starttime
                single_results['error'] = relativeerror
                single_results['data'] = analysisdata
                analysisdata.to_csv('output/' + mood_name + '_' + method + '_results.csv', index=True)
            result_db[mood_name][method] = single_results
            ntfy(method)
    # plotting data
    ntfy("Analysis!")
    ntfy("Performance")
    plotting_performance(result_db)
    plotting_prediction(result_db)
    reporting(result_db)
'''
    plotting_results(initial_dataframe, filtered, 'x')
    bestmethodname, filtered = filteringresults(result_db)
    plotting_results(initial_dataframe, filtered, 'x')
    plotting_results(initial_dataframe, filtered, 'b')
    reporting(result_db, bestmethodname)
    ntfy("Accomplished!")
'''