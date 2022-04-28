import skrf as rf
from skrf.media import Coaxial
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import os.path
import pandas as pd
from scipy.interpolate import interp1d
from skrf.constants import K_BOLTZMANN


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


### Added

###############################
##### Component Functions   ###
###############################
#(From Jihee's components.py and analyis_run1C_tsys.py)

def pow_to_dbm(x):
    return 10*np.log(x/1e-3)
def db_to_power_ratio(x):
    return 10**(x/10)

def dBm_to_W(p_dBm):
    return 10**(p_dBm/10)/1000
def power_in_dBm_per_Hz_to_temp(p):
    return dBm_to_W(p)/K_BOLTZMANN

### Function to read network in noisy environment ###
def read_in_noisy_network(ntw,freq_interp,Tphys):
    ntw_interp = ntw.interpolate(freq_interp, kind='cubic')
    ntw_interp_noisy = rf.NoisyNetwork(ntw_interp)
    ntw_interp_noisy.noise_source(source='passive',T0 = Tphys)

    return ntw_interp_noisy

### Function to make coaxial cable with coaxial line and SMA connector sample ###
def make_coax_cable(c_type,freq,len_m,Tphys):
        
    if c_type=='RG405U':
        ntw_coax_RT = cable_RG405U(freq,len_m)
        ntw_coax_cryo = RT2Cryo(ntw_coax_RT)
        # ntw_coax_cryo = ntw_coax_RT
        # ntw_SMA_RT = cmp.cable_RG405U(freq,0.04)
        ntw_SMA_RT = SMA_RG405U(freq,0.015)
        ntw_SMA_cryo = RT2Cryo(ntw_SMA_RT)
        # ntw_SMA_cryo = ntw_SMA_RT
    elif c_type=='PESR405FL':    
        ntw_coax_RT = cable_PESR405FL(freq,len_m)
        ntw_coax_cryo = RT2Cryo(ntw_coax_RT)
        # ntw_coax_cryo = ntw_coax_RT
        # ntw_SMA_RT = cmp.cable_PESR405FL(freq,0.04)
        ntw_SMA_RT = SMA_PESR405FL(freq,0.015)
        ntw_SMA_cryo = RT2Cryo(ntw_SMA_RT)
        # ntw_SMA_cryo = ntw_SMA_RT
    elif c_type=='NbTi085':
        ntw_coax_cryo = cable_NbTi085(freq,len_m)
        ntw_SMA_cryo = cable_NbTi085(freq,0.015)
    else:
        print('Error: Proper cable type is missing')

    noisy_ntw_coax = read_in_noisy_network(ntw_coax_cryo,freq,Tphys)
    noisy_ntw_SMA = read_in_noisy_network(ntw_SMA_cryo, freq, Tphys)
    
    assmb_cable = rf.MultiNoisyNetworkSystem()

    noisy_ntw_SMA.add_noise_polar(1e-6, 0.5)
    assmb_cable.add(noisy_ntw_SMA,'SMA1')
    assmb_cable.add(noisy_ntw_coax,'coax')
    noisy_ntw_SMA.add_noise_polar(1e-6, 0.55)
    assmb_cable.add(noisy_ntw_SMA,'SMA2')

    assmb_cable.connect('SMA1',2, 'coax',1)
    assmb_cable.connect('coax',2, 'SMA2',1)

    assmb_cable.external_port('SMA1',1,1)
    assmb_cable.external_port('SMA2',2,2)

    noisy_ntw_cable = assmb_cable.reduce()
    noisy_ntw_cable = read_in_noisy_network(noisy_ntw_cable,freq,Tphys)
        
    return noisy_ntw_cable 

## Paths to files - change to location of amplifier data

ZX60_33LN_NF_loc_PATH = r'./data/amplifier_data/ZX60-33LN-S+NF_DATA.csv'
AMP_PATH = './data/touchstone_files/amplifiers'
AMP_NAME = 'ZX60-33LNR-S+_UNIT1_.s2p' #Our amplifier is ZX60-33LN-S+, gain of LNR ~0.1 dB less at 1 Ghz


### Code from Jihee ###
# Creates a function of frequency that returns amplifier noise figure amplitudu
def gen_ZX60_33LN_S_NF_func(ZX60_33LN_NF_loc):
    
    # read in nf data from a csv
    # convert noise figure in db to amplitude
    # Make a new column with frequency in GHz
    # Interpolate noise figure as a function of frequency
    
    noise_figure_data2 = pd.read_csv(ZX60_33LN_NF_loc , header = 'infer')
    noise_figure_data2['NF Amplitude'] = 10**(noise_figure_data2['Noise Figure (5V)']/10)
    noise_figure_data2['Frequency (GHz)'] = noise_figure_data2['Frequency (MHz)']/1e3
    NF_Func2 = interp1d(noise_figure_data2['Frequency (GHz)'],noise_figure_data2['NF Amplitude'])#,fill_value=1000) # <- extrapolate ok here?
    
    return NF_Func2

def return_ZX60_33LN_params(Amp_path,Amp_name,freq,sat_check=False,vendor=True,Temp=290):
    
    # create amplifier from manufacturer's s2p file
    
    ZX60_33LN_NF_loc = ZX60_33LN_NF_loc_PATH
    
    Tphys = Temp
    
    Amp_loc = os.path.join(Amp_path,Amp_name)
    amp_sparam = rf.Network(Amp_loc)
    amp_sparam = amp_sparam.interpolate(freq,kind='cubic')
    amp_sparam  = rf.NoisyNetwork(amp_sparam)
    amp_sparam.noise_source(source='passive',T0=Tphys) # <-- why do we need this if we are specifying the cs matrix from the datasheet?
    
    # generate noise figure function
    
    #try:
    NF_func = gen_ZX60_33LN_S_NF_func(ZX60_33LN_NF_loc)
    nfig_amplitude = NF_func(freq.f/1e9)
    #except:
       # print('error')
       # return
    
    # use noise figure function to specify noise covariance matrix
    # cs}_ii = k_B T_0 S A S^dagger }_ii (F_i - 1)
        
    #Tnoise = rf.NetworkNoiseCov.Tnoise(freq.f,Tphys) # <-- where is this noise temperature used?
    
    ## Noise covariance matrix calculation 
    
    I = np.identity(np.shape(amp_sparam.s)[1])
    AS = np.matmul(I,np.conjugate(amp_sparam.s.swapaxes(1,2)))
    SAS = np.matmul(amp_sparam.s,AS)
    
    amp_sparam.cs[:,0,0] = (nfig_amplitude - 1) * K_BOLTZMANN * 290 * SAS[:,0,0]
    
    I = np.identity(np.shape(amp_sparam.s)[1])
    AS = np.matmul(I,np.conjugate(amp_sparam.s.swapaxes(1,2)))
    SAS = np.matmul(amp_sparam.s,AS)
    
    amp_sparam.cs[:,1,1] = (nfig_amplitude-1) * K_BOLTZMANN * 290 * SAS[:,1,1]
    
    amp_sparam.cs[:,0,1] = np.zeros_like(amp_sparam.cs[:,1,1])
    amp_sparam.cs[:,1,0] = amp_sparam.cs[:,0,1]
    
    return amp_sparam


def create_measured_circulator(freq,s_circ_db):
    
    
    s_circ = 10**(s_circ_db/10)

    s = np.zeros((len(freq),3,3), dtype ="complex_")
    
    #Assumes clockwise circulator (1=>2=>3=>1...)
    s[:,0,0] = s_circ[0,0]
    s[:,0,1] = s_circ[0,1]
    s[:,0,2] = s_circ[0,2]
    s[:,1,0] = s_circ[1,0]
    s[:,1,1] = s_circ[1,1]
    s[:,1,2] = s_circ[1,2]
    s[:,2,0] = s_circ[2,0]
    s[:,2,1] = s_circ[2,1]
    s[:,2,2] = s_circ[2,2]
    
    ntw_circ = rf.Network(frequency=freq, s=s)
    
    return ntw_circ

### code from Professor Rybka ###

#Generate a scikit-rf network for a cavity mode #
#with resonant frequency f0 and unloaded Q Q0
#and a list of coupling Qs for ports
def create_cavity_Qports(freq,f0,Q0,Qports,name='cavity'):
    
    Qports=np.array(Qports) #in case its a list    
    #Follows Ishikawa et al. IEICE Transactions on Electronics, E76-C Issue 6 Pages 925-931 (1993)
    Qomega=1/(1/Q0+np.sum(1/Qports)+1j*(freq.f/f0-f0/freq.f))    
    S=np.zeros([len(freq.f),len(Qports),len(Qports)],complex)
    
    #this could be made more elegant with np.outer but much less clear
    
    for i in range(len(Qports)):
        for j in range(i,len(Qports)):
            
            if i==j: #Diagonal S elements
                S[:,i,i]=2*Qomega/Qports[i]-1
                
            else: #off-diagonal S elements (symmetric)
                S[:,i,j]=2*Qomega/np.sqrt(Qports[i]*Qports[j])                        
                S[:,j,i]=S[:,i,j]
                
    ntwk = rf.Network(frequency=freq, s=S, name=name)
    return ntwk


def create_circulator_variable_leakage(freq,leakage,reflection):
    
    l = leakage
    g = reflection
    s_circ_db = np.array([[g,   l,-0.8],
                          [-0.8,g,   l],
                          [l, -0.8,  g]])
    
    
    s_circ = 10**(s_circ_db/10)

    s = np.zeros((len(freq),3,3), dtype ="complex_")
    
    #Assumes clockwise circulator (1=>2=>3=>1...)
    s[:,0,0] = s_circ[0,0]
    s[:,0,1] = s_circ[0,1]
    s[:,0,2] = s_circ[0,2]
    s[:,1,0] = s_circ[1,0]
    s[:,1,1] = s_circ[1,1]
    s[:,1,2] = s_circ[1,2]
    s[:,2,0] = s_circ[2,0]
    s[:,2,1] = s_circ[2,1]
    s[:,2,2] = s_circ[2,2]
    
    ntw_circ = rf.Network(frequency=freq, s=s)
    
    return ntw_circ