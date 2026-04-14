COMMENT
Acker, Corey D., Nancy Kopell, and John A. White. “Synchronization of Strongly 
Coupled Excitatory Neurons: Relating Network Behavior to Biophysics.” 
Journal of Computational Neuroscience 15, no. 1 (July 1, 2003): 71–90. 
https://doi.org/10.1023/A:1024474819512.
ENDCOMMENT

TITLE Stellate cells mechanism :acker


UNITS {
    (mV)=(millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
    
}

NEURON {
    SUFFIX fake_stellate_mech
    USEION na READ ena WRITE ina 
    USEION k READ ek WRITE ik 
    NONSPECIFIC_CURRENT il 
    NONSPECIFIC_CURRENT ih 
    RANGE gnabar,gkbar,gnap_bar,ghbar,ena,ek,mhf,mhs,gh,gna
    GLOBAL hf_tau_input,hs_tau_input
    

      
}

PARAMETER {
    gnap_bar = 0.0005 (S/cm2)
    gnabar = 0.052 (S/cm2) 
    gkbar = 0.011 (S/cm2)

    ghbar = 0.0015 (S/cm2)
    ena = 55 (mV) :reset by neuron, set after initialization
    ek = -90 (mV) :reset by neuron, set after initialization
    el = -65 (mV)
    gl = 0.0005 (S/cm2)
    eh = -10 (mV)
    hf_tau_input=0.51 (ms)
    hs_tau_input=5.6 (ms)






}   

ASSIGNED {
        v (mV)

	    gna (S/cm2)
	    gk (S/cm2)
        gh (S/cm2)
        gnap(S/cm2)
        ina (mA/cm2)
        ik (mA/cm2)
        il (mA/cm2)
        ih (mA/cm2)

    amna (1/ms) bmna (1/ms) ahna (1/ms) bhna (1/ms) ank (1/ms) bnk (1/ms) mnap_beta(1/ms) mnap_alpha (1/ms)
    mhfinf mhsinf  
    mhstau (ms)  mhftau (ms)   
} 


STATE {
    mna hna mnap nk mhf mhs 
}


BREAKPOINT { 
    SOLVE states METHOD cnexp

    gna = (gnabar*mna*mna*mna*hna)

    gnap=(gnap_bar*mnap)
	ina = (gna+gnap)*(v - ena)

    gk = (gkbar*nk*nk*nk*nk)
	ik = gk*(v - ek)      
    il = gl*(v - el)
    gh = ghbar*(0.65*mhf+0.35*mhs)
    ih = gh*(v-eh)
    



}
INITIAL {
    rates(v)
    mna = amna/(amna+bmna)
    hna = ahna/(ahna+bhna)
    mnap = mnap_alpha/(mnap_alpha+mnap_beta)

    nk = ank/(ank+bnk)

    mhf = mhfinf
    mhs = mhsinf
    

}

DERIVATIVE states {
    rates(v)

    mna' = amna*(1-mna) - bmna*mna
    hna' = ahna*(1-hna) - bhna*hna
    mnap' = mnap_alpha*(1-mnap) - mnap_beta*mnap
    nk' = ank*(1-nk) - bnk*nk

    mhf' = (mhfinf-mhf)/(mhftau)
    mhs' = (mhsinf-mhs)/(mhstau)
    


}
UNITSOFF
PROCEDURE rates(v (mV)) {
    amna = (.1)*vtrap(-(v+23),10)
    bmna = 4*exp(-(v+48)/18)
    ahna = 0.07*exp(-(v+37)/20)
    bhna = 1/(exp(-0.1*(v+7))+1)
    mnap_alpha=1/(0.15*(exp(-(v+38)/6.5)+1))
    mnap_beta = (exp(-(v+38)/6.5))/(0.15*(exp(-(v+38)/6.5)+1))
    ank = 0.01*vtrap(-(v+27),10)
    bnk = 0.125 * exp(-(v+37)/80)
    mhfinf = 1/(1+exp((v+79.2)/9.78))
    mhftau = ((hf_tau_input / ((exp((v-1.7)/10)) + exp(-(v+340)/52))) + 1) :0.51
    mhsinf = 1/(1+exp((v+71.3)/7.9))
    mhstau = ((hs_tau_input/ ((exp((v-1.7)/14)) + exp(-(v+260)/43))) + 1 ) :5.6
    
    


}

FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}

UNITSON
