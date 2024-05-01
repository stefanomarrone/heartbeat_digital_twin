import pandas as pd
from mako.template import Template
import subprocess


class FspnGenerator:
    def __init__(self, templatename):
        self.template = Template(filename=templatename)

    def filewriting(self, content, fspnfile):
        fhandle = open(fspnfile,'w')
        fhandle.write(content)
        fhandle.close()

    def executing(self, fspnfile):
        commandline = "Sim"
        p = subprocess.Popen([commandline,fspnfile], stdout=subprocess.DEVNULL)
        p.wait()

    def merge(self, df1, df2):
        dataset = pd.DataFrame(df1)
        dataset.insert(2, "b", df2.b)
        return dataset

    def filetotuplelist(self, fname):
        handler = open(fname)
        content = list(handler.readlines())
        content = list(map(lambda x: tuple(x.split(' ')), content))
        content = list(filter(lambda x: len(x) == 7, content))
        content = list(map(lambda x: (float(x[1]), float(x[3]), float(x[4])), content))
        content = list(map(lambda x: (x[0], (x[1] + x[2])/2), content))
        df = pd.DataFrame(content)
        handler.close()
        return df

    def extractingdata(self, xoutfilename, boutfilename):
        xdata = self.filetotuplelist(xoutfilename)
        xdata.columns = ['time', 'x']
        bdata = self.filetotuplelist(boutfilename)
        bdata.columns = ['time', 'b']
        fspndata = self.merge(xdata,bdata)
        return fspndata

    def execute(self, values):
        concrete = self.template.render(**values)
        self.filewriting(concrete, values['concretefile'])
        self.executing(values['concretefile'])
        data = self.extractingdata(values['x_plot'], values['b_plot'])
        return data




