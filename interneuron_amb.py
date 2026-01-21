#Interneuron class

from neuron import h
from cell import Cell


class Interneuron(Cell):
    name = "Interneuron"

    def _set_morphology(self):
        # create cells
        self.soma = h.Section(name="soma", cell=self)
        self.soma.L = 10 / 3.14
        self.soma.diam = 10  # surface_area= 100 um2

    def _set_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Cytoplasmic resistivity [ohm-cm]
            sec.cm = 1  # Specific capacitance [uf/cm2]

        # insert channels
        self.soma.insert("naf")
        self.soma.insert("kdr")
        self.soma.insert("pas")
    

        for seg in self.soma:
            seg.pas.g = 1e-4
            seg.pas.e = -65  # for leak conductance

        # add synaptic objects
        # exc synapse
        self.exc_syn = h.Exp2Syn(self.soma(0))
        self.exc_syn.tau1 = 0.78
        self.exc_syn.tau2 = 5.3
        self.exc_syn.e = 0

        # inhibitory synapse
        self.inhb_syn = h.Exp2Syn(self.soma(0))
        self.inhb_syn.tau1 = 8.3e-2
        self.inhb_syn.tau2 = 10
        self.inhb_syn.e = -75

    def _default_instrumentation(self):
        self.ext_dc = h.IClamp(self.soma(0.5))  # IClamp for const DC

        self.noise = h.IClamp(self.soma(0.5))  # IClamp for noise
        self.init_noise = h.IClamp(self.soma(0.5))  # IClamp for intial noise
        self.recorder ={}
        self.instr = {"IClamps":[],"IClamp_amps":[]}