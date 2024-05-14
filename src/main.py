import math
import time

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from fspns import FspnGenerator
from inference import InferencedHeart
from model import Heart, NoisyHeart
from moods import Mood


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


def plotting_results(odedf, fspndf, feature):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = odedf.plot(x='ode time', y='ode ' + feature)
    fspndf.plot(ax=ax, x='fspn time', y='fspn ' + feature)
    plt.savefig(feature + '.pdf')


def computeerror(expected, actual):
    globalerror = 0
    for k in expected.keys():
        error = (expected[k] - actual[k]) / expected[k]
        globalerror += math.sqrt(error ** 2)
    return globalerror


def minerror(x, y):
    retval = x
    if x[1] > y[1]:
        retval = y
    return retval


def ntfy(msg):
    requests.post("https://ntfy.sh/alerts-heartbeat-code", data=msg.encode(encoding='utf-8'))


def filteringresults(result_db):
    besterror = None
    bestmethod = None
    bestdata = None
    for k in result_db.keys():
        result = result_db[k]
        if not result['errorstate']:
            if bestmethod is None or besterror > result['error']:
                bestmethod, bestdata, besterror = k, result['data'], result['error']
    return bestmethod, bestdata


def reporting(general_db, methodname):
    reportname = 'report.txt'
    handler = open(reportname, 'w')
    handler.write("Best method = " + methodname + '\n*****\n')
    for key in general_db.keys():
        item = general_db[key]
        handler.write(key + '\n')
        handler.write("Error state = " + str(item['errorstate']) + '\n')
        handler.write("Inference time = " + str(item['inferencetime']) + '\n')
        if item['errorstate'] is False:
            handler.write("Error  = " + str(item['error']) + '\n')
            handler.write("Simulation time = " + str(item['simulationtime']) + '\n')
            handler.write("Report  = " + str(item['report']) + '\n')
            handler.write('*****\n\n')
    handler.close()


if __name__ == '__main__':
    # preparation
    ntfy("Starting")
    methods = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping', 'ampgo', 'nelder',
               'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs', 'tnc', 'trust-ncg', 'trust-exact',
               'trust-krylov', 'trust-constr', 'dogleg', 'slsqp', 'emcee', 'shgo', 'dual_annealing']
    seconds = 360
    x0 = 1
    b0 = 0
    template = 'heartbeat_template.fspn'
    # running initial model
    initial = (x0, b0)
    t, step = np.linspace(0, seconds, seconds * 50, retstep=True)
    modelled = modelling(initial, t, 'normal')
    initial_dataframe = pd.DataFrame(list(modelled))
    initial_dataframe.columns = ['ode x', 'ode b']
    initial_dataframe.insert(2, "ode time", t)
    initial_dataframe.to_csv('modelled.csv', index=True)
    result_db = dict()
    for method in methods:
        # inferencing the model from data
        print("starting: " + method)
        print("@" + time.ctime())
        starttime = time.time()
        inferenced, errorstate, report = inferencing(initial, t, modelled, method)
        stoptime = time.time()
        single_results = dict()
        single_results['report'] = report
        single_results['inferencetime'] = stoptime - starttime
        print("time = " + str(stoptime - starttime))
        single_results['errorstate'] = errorstate
        if not errorstate:
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
            starttime = time.time()
            analysisdata = modelgenerator.execute(makoparameters)
            stoptime = time.time()
            relativeerror = computeerror(mood.get('normal'), dict(inferenced))
            single_results['simulationtime'] = stoptime - starttime
            single_results['error'] = relativeerror
            single_results['data'] = analysisdata
            analysisdata.to_csv(method + '_results.csv', index=True)
        result_db[method] = single_results
        ntfy(method)
        print("ending: " + method + '\n')
    # plotting data
    ntfy("Finished!")
    bestmethodname, filtered = filteringresults(result_db)
    #plotting_results(initial_dataframe, filtered, 'x')
    #plotting_results(initial_dataframe, filtered, 'b')
    reporting(result_db, bestmethodname)
    ntfy("Accomplished!")
