from sklearn import tree
import pandas as pd
from six import StringIO
import pydotplus
from sklearn.tree import export_graphviz


class Mood:
    def __init__(self):
        self.database = {
            'normal': {'eps': 0.1, 'T': 1, 'xa': 0.1}
        }
        self.nominal = 'normal'

    def addmood(self, label, parameters):
        self.database[label] = parameters

    def moods(self):
        return self.database.keys()

    def trainclassfier(self):
        self.classifier = tree.DecisionTreeClassifier()
        features, target = self.getdata()
        self.classifier = self.classifier.fit(features, target)
        self.draw()

    def get(self, label):
        return self.database[label]

    def getdata(self):
        parameternames = self.database[self.nominal].keys()
        column = list(parameternames)
        column.append('mood')
        df = pd.DataFrame(columns=column)
        moods = self.database.keys()
        i = 0
        for mood in moods:
            df.at[i, 'mood'] = mood
            for parametername in parameternames:
                df.at[i, parametername] = self.database[mood][parametername]
            i += 1
        X = df[parameternames]
        y = df['mood']
        return X, y

    def getparameters(self, label):
        dictionary = self.database.get(label, self.nominal)
        return dictionary

    def draw(self):
        dot_data = StringIO()
        feature_cols = list(self.database[self.nominal].keys())
        target_names = list(self.database.keys())
        export_graphviz(self.classifier, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=feature_cols,
                        class_names=target_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf('output/moodclassifier.pdf')


    def getmostlikelymood(self, dictionary):
        parameternames = list(self.database[self.nominal].keys())
        df = pd.DataFrame(columns=parameternames)
        for parametername in parameternames:
            df.at[0, parametername] = dictionary[parametername]
        y_pred = self.classifier.predict(df)
        return y_pred
