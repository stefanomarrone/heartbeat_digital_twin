from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

class Mood:
    def __init__(self):
        self.database = {
            'normal': {'eps': 0.1, 'a': .5, 'xa': 1},
            # todo: search for parameters under coffee
            'coffee': {'eps': 0.001, 'a': .81, 'xa': .45}
        }
        self.nominal = 'normal'
        couples = self.getcouples()
        self.classifier = BayesianNetwork(couples)
        data = self.getdata()
        self.classifier.fit(data, estimator=MaximumLikelihoodEstimator)

    def getcouples(self):
        keys = self.database[self.nominal].keys()
        retval = list(map(lambda x: (x, 'mood'),keys))
        return retval


    def getdata(self):
        keys = list(self.database[self.nominal].keys())
        keys.append('mood')

        return None


    def getparameters(self, label):
        dictionary = self.database.get(label,self.nominal)
        return dictionary

    def getmlmood(self, dictionary):
        #todo you have to implement the method considering the construction of the Bayesian Network model
        pass