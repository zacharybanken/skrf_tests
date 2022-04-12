import skrf as rf
from skrf.media import Coaxial
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

### Ideal attenuator
def create_ideal_atten(freq, atten_dB):
    if atten_dB <0:
        raise ValueError("atten_dB must be a positive value")

    eps = 1e-32
    s = np.zeros((len(freq), 2, 2))

    s[:,0,0] = eps
    s[:,0,1] = 1
    s[:,1,0] = 1
    s[:,1,1] = eps

    # constructing Network object
    ntw_att = rf.Network(frequency=freq, s=s)
    
    atten = 10**(-atten_dB/20)
    ntw_att.s = atten*ntw_att.s
    return(ntw_att)

####### Coax (cable) lines (no connectors) ####### 

### Pasternack RG405U cable 
def cable_RG405U(freq, len_m):
    # Model for copper conductivity
    # Copper-clad steel: 40% of copper RT electrical conductivity
    sigma_cu = 0.4*58.5e6
    
    d_out = 0.0022098
    # Fitted value with Pasternack vendor data
    d_int = 1.06601197e-04
    epsilon = 3.52705118e-01 #expanded PTFE
    tand = 5.04329472e-02

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_cu)
    cable = coax.line(len_m,'m',embed=True)
    return cable

### SMA connector model from Sckikit-RF cable based on the Pasternack RG405U  
def SMA_RG405U(freq, len_m):
    # Model for copper conductivity
    # Copper-clad steel: 40% of copper RT electrical conductivity
    sigma_cu = .4*58.5e6
    
    d_out = 0.0022098
    # Fitted value with Pasternack vendor data
    d_int = 1.06601197e-04
    epsilon = 2e+0 # adjusted until matching better with the FNAL data (cable measurement RT)
    tand = 5.04329472e-02

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_cu)
    cable = coax.line(len_m,'m',embed=True)
    return cable


### Pasternack PE_SR405FL cable (Fitted data based on RT Pasternack data)
def cable_PESR405FL(freq, len_m):
    # Model for copper conductivity
    # Copper-clad steel: 40% of copper RT electrical conductivity
    sigma_cu = 0.4*58.5e6
    
    d_out = 0.002159
    # Fitted value with Pasternack vendor data
    d_int = 0.00120545
    epsilon = 1.16508632 #expanded PTFE
    tand = 0.01432541

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_cu)
    cable = coax.line(len_m,'m',embed=True)
    return cable

def SMA_PESR405FL(freq, len_m):
    # Model for copper conductivity
    # Copper-clad steel: 40% of copper RT electrical conductivity
    sigma_cu = 0.4*58.5e6
    
    d_out = 0.002159
    # Fitted value with Pasternack vendor data
    d_int = 0.00120545
    epsilon = 0.70e1 #expanded PTFE
    tand = 0.01432541

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_cu)
    cable = coax.line(len_m,'m',embed=True)
    return cable

### Keycom 0.085" NbTi NbTi cable (superconducting cable)  
# Used predicted coaxial cable attenuation data from Andrew Sonnenschein's (FNAL) slide 
def cable_NbTi085(freq, len_m):
    # Adjusted conductivity of NbTi 
    sigma_NbTi = 1e10

    d_out = 0.002159
    # Fitted value with Pasternack vendor data
    d_int = 6.75542915e-05
    epsilon = 5.10368260e+00 #expanded PTFE
    tand = 9.65943799e-06

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_NbTi)
    cable = coax.line(len_m,'m',embed=True)
    return cable

### Terminator from Pasternack RG405U cable spec
def create_terminator(freq, Gamma_0):
    # Model for copper conductivity
    # Copper-clad steel: 40% of copper RT electrical conductivity
    sigma_cu = 0.4*58.5e6
    
    d_out = 0.0022098
    # Fitted value with Pasternack vendor data
    d_int = 1.06601197e-04
    epsilon = 3.52705118e-01 #expanded PTFE
    tand = 5.04329472e-02

    coax = Coaxial(frequency=freq, z0=50, Dint=d_int, Dout= d_out, epsilon_r=epsilon, tan_delta=tand, sigma=sigma_cu)
    termination = coax.load(Gamma0=Gamma_0, nports=1)
    return termination


### SMA connector from HULL140A-29P-29P-120_sn3441 (120") SMA cable
def create_SMA(freq):
    # SMA_path = r'.\Data\Example Data'
    # SMA_name = r'HULL140A-29P-29P-120_sn3441.s2p'

    # SMA_file = str(Path(SMA_path)/SMA_name)
    # SMA_meas = rf.Network(SMA_file)

    ### changed due to file not found error (Jupyter on mac)
    SMA_file = r'./Data/Example Data/HULL140A-29P-29P-120_sn3441.s2p'
    SMA_meas = rf.Network(SMA_file)

    # we will focus on s11
    s11 = SMA_meas.s11

    #  time-gate the first largest reflection
    s11_gated = s11.time_gate(center=0.51, span=2)
    c = 3e8
    s21_gated = (1-abs(s11_gated.s[:,0,0]))*np.exp(-1j*(2*np.pi/(c/SMA_meas.frequency.f))*0.02)
    s11_gated.name='gated probe'

    s = np.zeros_like(SMA_meas.s)

    s[:,0,0] = s11_gated.s[:,0,0]
    s[:,0,1] = s21_gated
    s[:,1,0] = s21_gated
    s[:,1,1] = s11_gated.s[:,0,0]

    ntw_SMA = rf.Network()
    ntw_SMA.frequency = SMA_meas.frequency
    ntw_SMA.s = s
    ntw_SMA.z0=50
    
    ntw_SMA_interp = ntw_SMA.interpolate(freq, kind='cubic')

    return(ntw_SMA_interp)
   
### Ideal SMA connector from Amphenol webpage claim ####
# https://www.amphenolrf.com/connectors/sma-connectors.html
# VSWR (worst case)= 1.20 + .03 f (GHz) : DC-12.4 GHz
# Insertion loss= .06 √(f(GHz)) dB Max
def create_ideal_SMA(freq):
        
    VSWR = 1.2 + 0.03*freq.f/1e9
    RL_db = -20*np.log10((VSWR-1)/(VSWR+1))
    IL_db = 0.06*np.sqrt(freq.f/1e9)
    
    RL_gain = 10**(-RL_db/20)
    IL_gain = 10**(-IL_db/20)
    
    s = np.zeros((len(freq), 2, 2))

    s[:,0,0] = RL_gain
    s[:,0,1] = IL_gain
    s[:,1,0] = IL_gain
    s[:,1,1] = RL_gain

    # constructing Network object
    ntw_SMA = rf.Network(frequency=freq, s=s)
        
    return(ntw_SMA)

### Ideal SMA connector from Amphenol webpage claim ####
# https://www.amphenolrf.com/connectors/sma-connectors.html
# VSWR (worst case)= 1.20 + .03 f (GHz) : DC-12.4 GHz
# Insertion loss= .06 √(f(GHz)) dB Max
# Added phase to S12,S21
def create_ideal_SMA2(freq):
        
    VSWR = 1.2 + 0.03*freq.f/1e9
    RL_db = -20*np.log10((VSWR-1)/(VSWR+1))
    IL_db = 0.06*np.sqrt(freq.f/1e9)
    
    RL_gain = 10**(-RL_db/20)
    IL_gain = 10**(-IL_db/20)
    c = 3e8
    
    IL_gain_phAdded = (1-abs(RL_gain))*np.exp(-1j*(2*np.pi*freq.f/c)*0.2)    
    s = np.zeros((len(freq), 2, 2))

    s[:,0,0] = RL_gain
    s[:,0,1] = IL_gain_phAdded
    s[:,1,0] = IL_gain_phAdded
    s[:,1,1] = RL_gain
    
    # constructing Network object
    ntw_SMA = rf.Network(frequency=freq, s=s)
        
    return(ntw_SMA)

# Used scikit-rf cable model and tried to match with FNAL data after assembling cables
# Insertion loss= .06 √(f(GHz)) dB Max
def create_ideal_SMA3(freq):

    ntw_SMA = cable_RG405U(freq, 0.04)    
  
    return(ntw_SMA)


### Ideal Circulator (lossless) 
def create_ideal_circulator(freq):
    
    eps = 1e-32
    s = np.zeros((len(freq),3,3), dtype ="complex_")
    
    #Assumes clockwise circulator (1=>2=>3=>1...)
    s[:,0,0] = eps
    s[:,0,1] = eps
    s[:,0,2] = 1
    s[:,1,0] = 1
    s[:,1,1] = eps
    s[:,1,2] = eps
    s[:,2,0] = eps
    s[:,2,1] = 1
    s[:,2,2] = eps
    
    ntw_circ = rf.Network(frequency=freq, s=s)
    
    return ntw_circ

### Adjusted by comparing with QuinnStar (77K) test data for Run1C circulator
def create_ideal_circulator2(freq):
 
    eps = 1e-32 # ideal return loss, not matching with 77 K data (15 dB introduce oscillation) 
    IL = 0.949511 # ~0.45 dB insertion loss from QuinnStar data
    s = np.zeros((len(freq),3,3), dtype ="complex_")
    
    #Assumes clockwise circulator (1=>2=>3=>1...)
    s[:,0,0] = eps
    s[:,0,1] = eps
    s[:,0,2] = IL
    s[:,1,0] = IL
    s[:,1,1] = eps
    s[:,1,2] = eps
    s[:,2,0] = eps
    s[:,2,1] = IL
    s[:,2,2] = eps
    
    ntw_circ = rf.Network(frequency=freq, s=s)
    
    return ntw_circ

### Ideal directional coupler (lossless, all matched)
def create_ideal_dirCoupler(freq, coupl_db):
    eps = 1e-32
    beta = 10**(-coupl_db/20)
    s = np.zeros((len(freq),4,4), dtype ="complex_")
    alpha = math.sqrt(1-beta**2)
        
    s[:,0,0] = eps
    s[:,0,1] = alpha
    s[:,0,2] = beta*1j  
    s[:,0,3] = eps
    s[:,1,0] = alpha
    s[:,1,1] = eps
    s[:,1,2] = eps
    s[:,1,3] = beta*1j
    s[:,2,0] = beta*1j 
    s[:,2,1] = eps
    s[:,2,2] = eps
    s[:,2,3] = alpha
    s[:,3,0] = eps
    s[:,3,1] = beta*1j
    s[:,3,2] = alpha
    s[:,3,3] = eps
    
    ntw_dirCpl = rf.Network(frequency=freq, s=s)
    
    return ntw_dirCpl

### Adjusted Insertion Loss (IL) with the FNAL measurement value (average -0.16 dB) for Run1C directional coupler (PE2208-20)    
def create_ideal_dirCoupler2(freq, coupl_db):
    eps = 1e-32
    # RL = 0.035
    beta = 10**(-coupl_db/20)
    IL = 0.981183 # -0.165 dB
    s = np.zeros((len(freq),4,4), dtype ="complex_")
        
    s[:,0,0] = eps
    s[:,0,1] = IL
    s[:,0,2] = beta*1j  
    s[:,0,3] = eps
    s[:,1,0] = IL
    s[:,1,1] = eps
    s[:,1,2] = eps
    s[:,1,3] = beta*1j
    s[:,2,0] = beta*1j 
    s[:,2,1] = eps
    s[:,2,2] = eps
    s[:,2,3] = IL
    s[:,3,0] = eps
    s[:,3,1] = beta*1j
    s[:,3,2] = IL
    s[:,3,3] = eps
    
    ntw_dirCpl = rf.Network(frequency=freq, s=s)
    
    return ntw_dirCpl

### Cryo adjustment for the cable insertion loss. (300 K -> 4 K (mK) for Cu clad steel)
# from bluefors loss in dB/m is roughly halved for regular coax cables below 4K 
# https://bluefors.com/products/coaxial-wiring/#technical-specifications 
# Attenuation (RT): 3.0 dB/m (1.0 GHz); 9.5 dB/m (10 GHz) 
# Attenuation (4K): 1.5 dB/m (1.0 GHz); 4.6 dB/m (10 GHz) 
# going to adjust for that by reducing losses by half (near term) someday need to make a cold cable model. (from Maurio's code)

def RT2Cryo(nwk):

    rate = (0.07/9)*((nwk.f/1e9)-1) + 2.0

    newgaindb_01 = 20*np.log10(abs(nwk.s[:,0,1]))/rate
    newgaindb_10 = 20*np.log10(abs(nwk.s[:,1,0]))/rate

    newgain_lin_01 = 10**(newgaindb_01/20)
    newgain_lin_10 = 10**(newgaindb_10/20)

    nwk_10_norm = nwk.s[:,1,0] / np.abs(nwk.s[:,1,0])
    nwk_01_norm = nwk.s[:,0,1] / np.abs(nwk.s[:,0,1])
    
    nwk.s[:,1,0] = nwk_10_norm*newgain_lin_10
    nwk.s[:,0,1] = nwk_01_norm*newgain_lin_01

    return nwk


