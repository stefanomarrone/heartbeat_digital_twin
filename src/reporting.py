import math
from matplotlib import pyplot as plt
from statistics import mean
import pandas as pd




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



def plotting_results(odedf, fspndf, feature, name, folder):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = odedf.plot(x='ode time', y='ode ' + feature)
    fspndf.plot(ax=ax, x='fspn time', y='fspn ' + feature)
    L = plt.legend()
    L.get_texts()[0].set_text('ODE')
    L.get_texts()[1].set_text('FSPN')
    plt.xlabel('Time [secs]')
    plt.ylabel(feature)
    plt.savefig(folder + name + '.pdf')
    plt.close()


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


def inner_reporting(d, feature, xlabel, ylabel, filename):
    moods = d.keys()
    times = dict()
    for moo in moods:
        for meth in d[moo].keys():
            times[meth] = list()
    for moo in moods:
        for meth in d[moo].keys():
            if not d[moo][meth]['errorstate']:
                times[meth].append(d[moo][meth][feature])
    for meth in list(times.keys()):
        if len(times[meth]) == 0:
            del times[meth]
    for meth in times.keys():
        times[meth] = mean(times[meth])
    courses = list(times.keys())
    values = list(times.values())
    labels = [str(v) for v in values]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xscale('log')
    ax.barh(courses, values, color='blue', height= 0.3)
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.savefig(filename)
    plt.close()


def reporting(general_db, folder):
    reportname = folder + 'report.txt'
    handler = open(reportname, 'w')
    for key in general_db.keys():
        for kkey in general_db[key].keys():
            item = general_db[key][kkey]
            handler.write(key + '\n')
            handler.write(kkey + '\n')
            handler.write("Error state = " + str(item['errorstate']) + '\n')
            handler.write("Inference time = " + str(item['inferencetime']) + '\n')
            if item['errorstate'] is False:
                handler.write("Most Probable Mood = " + str(item['recognized']) + '\n')
                handler.write("Error  = " + str(item['error']) + '\n')
                handler.write("Simulation time = " + str(item['simulationtime']) + '\n')
                handler.write("Report  = " + str(item['report']) + '\n')
            handler.write('*****\n\n')
    handler.close()

def plotting_prediction(d, folder):
    inner_reporting(d, 'error', "Algorithms", "Relative Error [%]", folder + 'prediction.pdf')


def plotting_performance(d, folder):
    inner_reporting(d, 'inferencetime', "Algorithms", "Inference Time [secs]", folder + 'performance.pdf')


def best_retrieve(d):
    mood, method = (None, None)
    error = None
    for moo in d.keys():
        for meth in d[moo].keys():
            if d[moo][meth]['errorstate'] == False:
                currenterror = d[moo][meth]['error']
                if error is None or error > currenterror:
                    error = currenterror
                    mood, method = moo, meth
    return mood, method, d[mood][method]['data']


def plotting_odes(db, feature, folder):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    for mood in db.keys():
        data = db[mood]
        renamed = pd.DataFrame(data)
        renamed.columns = renamed.columns.str.replace('ode ' + feature, mood)
        renamed.plot(ax=ax, x='ode time', y=mood)
    plt.xlabel('Time [secs]')
    plt.ylabel(feature)
    plt.savefig(folder + 'ode_moods_' + feature + '.pdf')
    plt.close()


