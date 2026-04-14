#stellate class

from neuron import h
from cell import Cell

class Stellate(Cell):
    name = "StellateCell"

    def _set_morphology(self):
        # Create cell
        self.soma = h.Section(name="soma", cell=self)
        self.soma.L = 10 / 3.14  # length
        self.soma.diam = 10  # Surface_area= 100 um2

    def _set_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Cytoplasmic resistivity [ohm-cm]
            sec.cm = 1.5  # Specific capacitance [uf/cm2]

        # add channels
        #self.soma.insert("i_theta_stell")
        self.soma.insert("fake_stellate_mech")

        self.soma.ena = 55  # if not set here, neuron uses default values.
        self.soma.ek = -90  # if not set here, neuron uses default values.
        # add synapses
        # exc
        self.exc_syn = h.Exp2Syn(self.soma(0))
        self.exc_syn.tau1 = 0.78
        self.exc_syn.tau2 = 5.3
        self.exc_syn.e = 0

        # inhib
        self.inhb_syn = h.Exp2Syn(self.soma(0))
        self.inhb_syn.tau1 = 8.3e-2
        self.inhb_syn.tau2 = 10
        self.inhb_syn.e = -75


    def _default_instrumentation(self):
        self.ext_dc = h.IClamp(self.soma(0.5))  # IClamp for const dc
        self.noise = h.IClamp(self.soma(0.5))  # IClamp for noise
        self.init_noise = h.IClamp(self.soma(0.5))  # IClamp for intial noise
        self.recorder = {}
        self.instr = {"IClamps":[],"IClamp_amps":[]}