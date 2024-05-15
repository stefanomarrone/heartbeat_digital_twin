import math
from matplotlib import pyplot as plt
from statistics import mean
import pandas as pd

def plotting_performance(d):
    moods = d.keys()
    times = dict()
    for moo in moods:
        for meth in d[moo].keys():
            times[meth] = list()
    for moo in moods:
        for meth in d[moo].keys():
            if d[moo][meth]['errorstate'] == False:
                times[meth].append(d[moo][meth]['inferencetime'])
    for meth in list(times.keys()):
        if len(times[meth]) == 0:
            del times[meth]
    for meth in times.keys():
        times[meth] = mean(times[meth])
    courses = list(times.keys())
    values = list(times.values())
    plt.bar(courses, values, color='blue', width=0.4)
    plt.xlabel("Algorithms")
    plt.ylabel("Inference time [secs]")
    plt.savefig('output/performance.pdf')



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



def plotting_results(odedf, fspndf, feature):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = odedf.plot(x='ode time', y='ode' + feature)
    fspndf.plot(ax=ax, x='fspn time', y='fspn ' + feature)
    L = plt.legend()
    L.get_texts()[0].set_text('ODE')
    L.get_texts()[1].set_text('FSPN')
    plt.xlabel('Time [secs]')
    plt.ylabel(feature)
    plt.savefig('output/' + feature + '.pdf')


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


def reporting(general_db):
    reportname = 'output/report.txt'
    handler = open(reportname, 'w')
    for key in general_db.keys():
        for kkey in general_db[key].keys():
            item = general_db[key][kkey]
            handler.write(key + '\n')
            handler.write(kkey + '\n')
            handler.write("Error state = " + str(item['errorstate']) + '\n')
            handler.write("Inference time = " + str(item['inferencetime']) + '\n')
            if item['errorstate'] is False:
                handler.write("Error  = " + str(item['error']) + '\n')
                handler.write("Simulation time = " + str(item['simulationtime']) + '\n')
                handler.write("Report  = " + str(item['report']) + '\n')
            handler.write('*****\n\n')

    handler.close()

def plotting_prediction(d):
    moods = d.keys()
    times = dict()
    for moo in moods:
        for meth in d[moo].keys():
            times[meth] = list()
    for moo in moods:
        for meth in d[moo].keys():
            if d[moo][meth]['errorstate'] == False:
                times[meth].append(d[moo][meth]['error'])
    for meth in list(times.keys()):
        if len(times[meth]) == 0:
            del times[meth]
    for meth in times.keys():
        times[meth] = mean(times[meth])
    courses = list(times.keys())
    values = list(times.values())
    plt.bar(courses, values, color='blue', width=0.4)
    plt.xlabel("Algorithms")
    plt.ylabel("Relative error")
    plt.savefig('output/prediction.pdf')

