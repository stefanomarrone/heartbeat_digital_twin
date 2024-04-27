class Mood:
    def __init__(self):
        self.database = {
            'normal': {'eps': 0.1, 'a': .5, 'xa': 1},
            # todo: search for parameters under coffee
            'coffee': {'eps': 0.001, 'a': .81, 'xa': .45}
        }
        self.nominal = 'normal'

    def getparameters(self, label):
        dictionary = self.database.get(label,self.nominal)
        return dictionary

    def adddictionary(self, newlabel, dictionary):
        self.database[newlabel] = dictionary
        return dictionary


    def getmlmood(self, dictionary):
        #todo you have to implement the method considering the construction of the Bayesian Network model
        pass