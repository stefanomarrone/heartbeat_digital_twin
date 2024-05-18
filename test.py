import time
import numpy as np
import pandas as pd
import requests
from src.fspns import FspnGenerator
from src.inference import InferencedHeart
from src.model import Heart, NoisyHeart
from src.moods import Mood
from src.reporting import computeerror, plotting_results, plotting_performance, plotting_prediction, reporting, \
    best_retrieve, plotting_odes


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


def ntfy(msg):
    requests.post("https://ntfy.sh/alerts-heartbeat-code", data=msg.encode(encoding='utf-8'))
    print(msg)

def influenceanalysis():
    conditionrepo = Mood()
    conditionrepo.addmood('normal', {'eps': 0.5, 'T': 1, 'xa': 0.1})
    conditionrepo.addmood('coffee', {'eps': 0.3, 'T': 1.5, 'xa': 0.1})
    conditionrepo.addmood('sleeping', {'eps': 1, 'T': 0.5, 'xa': 0.01})
    conditionrepo.trainclassfier()
    return conditionrepo

'''
    methodlist = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping', 'ampgo', 'nelder',
                  'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs', 'tnc', 'trust-ncg', 'trust-exact', 'trust-krylov',
                  'trust-constr', 'dogleg', 'slsqp', 'emcee', 'shgo', 'dual_annealing']
    methodlist = ['leastsq', 'least_squares', 'powell']
'''

def getMethods():
    if True:
        methodlist = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping', 'ampgo', 'nelder',
                  'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs', 'tnc', 'trust-ncg', 'trust-exact', 'trust-krylov',
                  'trust-constr', 'dogleg', 'slsqp', 'emcee', 'shgo', 'dual_annealing']
    else:
        methodlist = ['powell']
    return methodlist



if __name__ == '__main__':
    # preparation
    seconds = 25
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
    ode_db = dict()
    for mood_name in mood.moods():
        result_db[mood_name] = dict()
        modelled = modelling(initial, t, mood_name)
        initial_dataframe = pd.DataFrame(list(modelled))
        initial_dataframe.columns = ['ode x', 'ode b']
        initial_dataframe.insert(2, "ode time", t)
        ode_db[mood_name] = initial_dataframe
        initial_dataframe.to_csv('output/' + mood_name + '.csv', index=True)
        for method in methods:
            # inferencing the model from data
            starttime = time.time()
            inferenced, errorstate, report = mood.getparameters(mood_name), False, ''
            stoptime = time.time()
            single_results = dict()
            single_results['report'] = report
            single_results['inferencetime'] = stoptime - starttime
            single_results['errorstate'] = errorstate
            if not errorstate:
                mostprobable = mood_name
                # executing FSPN model
                makoparameters = dict(inferenced)
                makoparameters['seconds'] = seconds
                makoparameters['x0'] = x0 + 500
                makoparameters['b0'] = b0 + 500
                makoparameters['b_plot'] = "test/b_plot.out"
                makoparameters['x_plot'] = "test/x_plot.out"
                makoparameters['concretefile'] = 'test/heartbeat_temp.fspn'
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
                analysisdata.to_csv('test/' + mood_name + '_' + method + '_results.csv', index=True)
                plotting_results(initial_dataframe, analysisdata, 'x', mood_name + '_' + method + '_x', 'test/')
                plotting_results(initial_dataframe, analysisdata, 'b', mood_name + '_' + method + '_b', 'test/')
            result_db[mood_name][method] = single_results
            ntfy(method)
    # plotting data
    ntfy("Analysis!")
    ntfy("Performance")
    plotting_performance(result_db, 'test/')
    plotting_prediction(result_db, 'test/')
    reporting(result_db, 'test/')
    best_mood, bestmethodname, filtered = best_retrieve(result_db)
    plotting_results(ode_db[best_mood], filtered, 'x', 'time_best_x', 'test/')
    plotting_results(ode_db[best_mood], filtered, 'b', 'time_best_b', 'test/')
    plotting_odes(ode_db, 'x', 'test/')
    plotting_odes(ode_db, 'b', 'test/')
    print("Best mood: " + best_mood)
    print("Best method: " + bestmethodname)
    ntfy("Closing everything!")
    ntfy("Accomplished!")
