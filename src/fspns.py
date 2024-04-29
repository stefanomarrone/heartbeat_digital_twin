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

    def filetotuplelist(fname):
        fhandle = open(fname)
        content = list(fhandle.readlines())
        content = list(map(lambda x: tuple(x.split(' ')), content))
        fhandle.close()
        return content

    def extractingdata(self, xoutfilename, boutfilename):
        xdata = self.filetotuplelist(xoutfilename)
        bdata = self.filetotuplelist(boutfilename)

    def execute(self, values):
        concrete = self.template.render(**values)
        self.filewriting(concrete, values['concretefile'])
        self.executing(values['concretefile'])
        data = self.extractingdata(values['x_plot'], values['b_plot'])
        return data




