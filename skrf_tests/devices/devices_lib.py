# =============================================================================
# Electronic Tuning and Coupling Simulations
# =============================================================================

import skrf as rf
#from pylab import *
#rf.stylely()
from skrf import Frequency
#from skrf.media import mline
#from numpy import real, log10, sum, absolute, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import time 
import pickle
#import math
from scipy import interpolate as sc_interp
import tools_lib as tool
from pathlib import Path
import math
import cmath as cmt
from skrf.constants import K_BOLTZMANN


# =============================================================================
# Functions
# =============================================================================

#################################################################################
############## get and save combiners (high to low level) #######################
#################################################################################


############## open combiner or make one ##############

### Ideal
def get_ideal_combiner(N):
    
    directory = Path("./saved_combiners/ideal/")
    file_name = "saved_"+str(N)+"_port_ideal_combiner"
    full_file_extension = directory / file_name
    
    try:
        pickle_in = open(full_file_extension+".pickle","rb")
        print('tried to open')
    except: 
        print("Didn't have that file... making one now.")
        pickle_ideal_combiner(N,full_file_extension)
        pickle_in = open(full_file_extension+".pickle","rb")
    finally:
        combiner = pickle.load(pickle_in)
        
    return combiner

### WU
def get_WU_v2_combiner(N):
    
    directory = Path("./saved_combiners/WU_v2/")
    file_name = "saved_"+str(N)+"_port_WUv2_combiner"
    full_file_extension = directory / file_name
    
    try:
        pickle_in = open(full_file_extension+".pickle","rb")
    except: 
        print("Didn't have that N = "+str(N)+" file... making one now.")
        pickle_WU_v2_combiner(N,full_file_extension)
        pickle_in = open(full_file_extension+".pickle","rb")
    finally:
        combiner = pickle.load(pickle_in)
    
    return combiner



############## make then pickle combiner ##############

### Ideal    
def pickle_ideal_combiner(N,file_extension):
    
    PD2 = create_ideal_2port_combiner()
    combiner = Nway_combiner(N,PD2)
    pickle_object(combiner,file_extension)
    
### WU    
def pickle_WU_v2_combiner(N,file_extension):
       
    PD4 = get_WU_4port_combiner('WU_v2')
    combiner = Nway_combiner_base4(N,PD4)
    pickle_object(combiner,file_extension)
    

############## pickle any object ##############

def pickle_object(thing,file_extension):
    
    #directory = Path("/Users/bout736/Desktop/N-Cavities/Ncavity_Python_Modeling/saved_combiners/ideal/")
    #file_name = "saved_"+str(N)+"_port_ideal_combiner"
    print("Just called pickle_object()") 
    pickle_out = open(file_extension+".pickle","wb")
    pickle.dump(thing, pickle_out)
    pickle_out.close()
    
############## stack blocks ##############

### starting with 3 way combiner
def Nway_combiner(ways, Combiner):

    N = ways - 1
    for i in range(0,N):
        if i == 0:
            PD_out = Combiner
        else:
            PD_out = rf.network.connect(PD_out,1,Combiner,0)
    return PD_out

### starting with 3 way combiner
def Nway_combiner_base4(ways, Combiner):

    N = int(ways/3)
    for i in range(0,N):
        if i == 0:
            PD_out = Combiner
        else:
            PD_out = rf.network.connect(PD_out,1,Combiner,0)
    return PD_out

############## Get simplest building block ##############

### Ideal wilkinson combiner
def create_ideal_2port_combiner():
    
    #PD_ideal  = rf.Network(Path('/Users/bout736/Desktop/')) / 'BP2G1+_Plus25DegC_Unit1.S3P')
    PD_ideal  = rf.Network(str(Path('./data/touchstone_files/') / 'BP2G1+_Plus25DegC_Unit1.S3P'))

    #eps = 1e-31
    eps = 1e-32

    PD_ideal.s[:,0,0] = eps*np.ones_like(PD_ideal.s[:,0,0])
    PD_ideal.s[:,1,0] = (1)*np.ones_like(PD_ideal.s[:,1,0])
    PD_ideal.s[:,2,0] = (1)*np.ones_like(PD_ideal.s[:,2,0])
    PD_ideal.s[:,0,1] = (1)*np.ones_like(PD_ideal.s[:,0,1])
    PD_ideal.s[:,0,2] = (1)*np.ones_like(PD_ideal.s[:,0,2])
    PD_ideal.s[:,1,1] = eps*np.ones_like(PD_ideal.s[:,1,1])
    PD_ideal.s[:,2,2] = eps*np.ones_like(PD_ideal.s[:,2,2])
    PD_ideal.s[:,2,1] = eps*np.ones_like(PD_ideal.s[:,2,1])
    PD_ideal.s[:,1,2] = eps*np.ones_like(PD_ideal.s[:,1,2])
    
    #x,y,z = np.shape(PD_ideal.s)
    #print(x,y,z)
    #for i in range(0,x):
    #    PD_ideal.s[i,:,:] = (-1j/np.sqrt(2))*(PD_ideal.s[i,:,:])
    #print(PD_ideal.s[1,:,:])
    
    PD_ideal.s = (-1j/np.sqrt(2))*PD_ideal.s
    return(PD_ideal)



### WU
def get_WU_4port_combiner(version):
     
    fileName_ext = str(version)+'.s5p'
    PD_test  = rf.Network(str(Path('./data/touchstone_files/') / fileName_ext))
    return PD_test 


### Ideal attenuator
def create_ideal_atten(freq, atten_dB):
    eps = 1e-32
    s = np.zeros((len(freq), 2, 2))

    s[:,0,0] = eps
    s[:,0,1] = 1
    s[:,1,0] = 1
    s[:,1,1] = eps

    # constructing Network object
    ntw_att = rf.Network(frequency=freq, s=s)
    
    np_to_dB = 8.685889638
    ntw_att.s = (math.exp(-atten_dB/np_to_dB))*ntw_att.s
    return(ntw_att)

### Ideal cable

#Used Pasternack RG402 coax cable data
def create_ideal_cable(l_cable):
   
    cable_ideal = rf.Network(str(Path('./data/touchstone_files/') / 'P1_ATTEN2_P2_ATTEN1.S2P'))

    eps = 1e-32

    cable_ideal.s[:,0,0] = eps*np.ones_like(cable_ideal.s[:,0,0])
    cable_ideal.s[:,1,0] = (1)*np.ones_like(cable_ideal.s[:,1,0])
    cable_ideal.s[:,0,1] = (1)*np.ones_like(cable_ideal.s[:,0,1])
    cable_ideal.s[:,1,1] = eps*np.ones_like(cable_ideal.s[:,1,1])
    
    i = 0
    for freq in cable_ideal.f:
        c = 299792458 #speed of light in m/s
        b = 2*math.pi*freq/c
        #Line attenuation from powerlaw fitting of RG402(paskernack) data
        a = 15.05264 * freq/1e9 - 0.25607 * ((freq/1e9)**2) + 20.11384 
        np_to_dB = 8.685889638 #neper to dB conversion
        a = a/(np_to_dB*100) #in unit of neper/m
        cable_ideal.s[i] = cable_ideal.s[i]*(cmt.exp((-a + 1j*b)*l_cable))   
        i = i + 1
        
    return(cable_ideal)

#################### wrapper functions  #######################

def run_func_x_times(x_times, func, *args):
    
    t0 = time.time()
    t = 0
    output = []
    
    for i in range(x_times) :
        
        t = time.time()-t0
        output.append(func(*args))
        
    return output


def run_func_for_a_bit(computing_time, func, *args):
    
    t0 = time.time()
    t = 0
    output = []
    
    while t < computing_time:
        
        t = time.time()-t0
        output.append(func(*args))
        
    return output

    

################ Higher level functions creating interesting combiners ##################
  
def input_random_errors_to_all_combiner_channels(combiner, pow_sigma, phase_sigma): 
    
    ohms = 50

    a = input_stimulus_template(combiner)
    N = len(a)-1
    
    phase = np.random.normal(0,phase_sigma, N)
    pow_frac = np.random.normal(1,pow_sigma, N)
    for i, p in enumerate(pow_frac): 
        if p < 0:
            #print("p: "+str(p), phase[i], phase[i] + 180)
            pow_frac[i] = np.abs(p)
            phase[i] = phase[i] + 180
            

    #count, bins, ignored = plt.hist(s, 100, normed = True)
    #print(pow_frac)
    for i in range(1,len(a)):
        a[i] = complex_number(pow_frac[i-1], phase[i-1],amp=True)* np.sqrt(ohms) # don't forget to multiply by sqrt impedance
        #a[i] = complex_number(pow_frac[i+1], phase[i+1])* np.sqrt(ohms) # don't forget to multiply by sqrt impedance

    slice_number = 0
    
    power_in_each_port = np.absolute(a)
    power_in_each_port = np.divide(power_in_each_port,np.sqrt(ohms))
    power_in_each_port = np.multiply(power_in_each_port,power_in_each_port)
    total_power_in = np.sum( power_in_each_port )
    
    combiner_single_freq = get_slice(combiner,slice_number)
    
    M = get_M(combiner_single_freq , a ) 
    Normalized_M = M /total_power_in  #float(N) 

    return Normalized_M             


def give_all_combiners_random_phases(combiner, sigma): 
    
    power_fraction = 1
    ohms = 50
    mu = 0
    a = input_stimulus_template(combiner)
    
    s = np.random.normal(mu,sigma, len(a)-1)
    #count, bins, ignored = plt.hist(s, 100, normed = True)

    for i in range(1,len(a)):
        a[i] = complex_number(power_fraction, s[i-1])* np.sqrt(ohms) # don't forget to multiply by sqrt impedance
     
    slice_number = 0
    combiner_single_freq = get_slice(combiner,slice_number)
    
    M = get_M(combiner_single_freq , a ) 

    return M


####### returns 2^n or 4^n arrays #######

def valid_N(generations = None):
    
    if generations is None: generations = 12
    N_array = np.zeros(generations,dtype=int)
    for i in range(generations):N_array[i] = 2**(i+1)
    return N_array

def valid_WU_N(generations = None):
    
    if generations is None: generations = 6
    N_array = np.zeros(generations,dtype=int)
    for i in range(generations):N_array[i] = 4**(i+1)
    return N_array



def perturb_one_combiner_input(combiner, fraction_of_power, phase): 
    
    
    ohms = 50
    a = input_stimulus_template(combiner) 
    a[1] = complex_number(fraction_of_power, phase)* np.sqrt(ohms) # don't forget to multiply by sqrt impedance
     
    slice_number = 80
    combiner_single_freq = get_slice(combiner,slice_number)
    
    M = get_M(combiner_single_freq , a ) 

    return M


def full_band_combiner_response_to_ideal_input(combiner): 
    
    M_array = []
    a = input_stimulus_template(combiner) 
    power_in = len(a)-1

    
    for i in range(len(combiner)):
        combiner_single_freq = get_slice(combiner,i)
        M = get_M(combiner_single_freq , a )/power_in
        M_array.append(M)
    return M_array



####### picks out a particular freq band  #######

def get_slice(combiner, Nth_slice):
    
    N_slices = np.shape(combiner.s[:,:,:])[0]
    middle = int(np.floor(N_slices/2))
    return combiner.s[Nth_slice,:,:] 
    

####### takes freq slice, dots with stimulus and computes power  #######
    
def get_M(S_slice, a):
    
    ohms = 50
    combiner_slice_dotted_with_a = S_slice.dot(a) 
    combiner_slice_dotted_with_a = np.abs(combiner_slice_dotted_with_a)  
    M = combiner_slice_dotted_with_a**2 /ohms 
    power_out_of_output = M[0]
    return power_out_of_output
    
    
####### returns stimulus of 1s that is correct length  #######
   
def input_stimulus_template(network):
    ohms = 50   
    N_channels = np.shape(network.s)[1]
    array = np.ones(N_channels,dtype=complex)
    array = array * np.sqrt(ohms) # don't forget to multiply by sqrt impedance
    array[0] = 0 # don't send power in the output 
    return array


####### creates complex number from power (or amplitude) and phase #######

def complex_number(power_or_amp_fraction, phase, amp = False):
    
    degrees_to_radians = np.pi/180
    amplitude = np.sqrt(power_or_amp_fraction)
    if amp: amplitude = power_or_amp_fraction          
    complex_number_on_unit_circle = np.exp(1j*degrees_to_radians*phase)
    return amplitude * complex_number_on_unit_circle
    

########################### simple helping functions ###########################
    
def pow_to_dB(power_ratio): return 10*np.log10(power_ratio)
def amp_to_dB(amplitude_ratio): return 20*np.log10(amplitude_ratio)
def dB_to_pow(dB): return 10**(dB/10)
def dB_to_amp(dB): return 10**(dB/20)

def pow_frac_to_scan_rate(pow_frac): return pow_frac**2




########################### make cavities ###########################


def clone_cavity(cavity_s2p):
    
    
    cavity_s2p.plot_s_db(m=0,n=0,label='Real S11')


    freq = cavity_s2p.f
    S11 = cavity_s2p.s_db[:,0,0]
    
    center_index =  tool.find_nearest(S11, min(S11))
    center_freq = freq[center_index]
    points = len(freq)
    R_cavity=2000 #2000 #.75
    Q_cavity=2000
    N1 = 7
    N2 = .9
    
    
    path= Path('./touchstone_files/cavities/')

    print(freq[0], center_freq)
    
    model = make_cavity__with_antennas_network(freq[0],center_freq, freq[-1], N1 = N1, N2 = N2, R = R_cavity, Q = Q_cavity, points = points)
    model.plot_s_db(m=0,n=0,label='Real S11')

    
    
    dic = {'model':model,'freq':freq, 'S11':S11, 'center':center_freq}
    return(dic)



def make_cavity__with_antennas_network(start_freq, mode_freq, stop_freq, Q = 10000.0, R = 1.0, N1 = 1, N2 = 1, points = 10001):
    
    cav = make_cavity_network(start_freq,mode_freq, stop_freq, Q = Q, R = R, points = points) # , plot=True , save=True
    port_N2 = make_antenna_network(N2,start_freq, stop_freq,points = points)
    port_N1 = make_antenna_network(N1,start_freq, stop_freq,points = points)
    
    ### Add one antenna to cavity 
    cavity_ant = rf.network.connect(cav,1,port_N2,0)
    ### Add the other antenna to cavity 
    ant_cavity_ant = rf.network.connect(port_N1,0,cavity_ant,1)
    return(ant_cavity_ant)

    
    

def make_cavity_network(start_freq, mode_freq, stop_freq, Q = 10000.0, R = 1.0, points = 10001): # , plot = False, save = False
    

    # in GHZ 
    
    #cavity parameters  
    
    #R=1
    #Q=10e3
    #mode_freq = mode_freq*1e9
    mode_freq = mode_freq

    #C = .01e-12    
    #L = calc_L(mode_freq,C)
    Z0 = 50
    L,C = calc_L_and_C(mode_freq, R , Q )
    
    #points=20001
    #pts=200001
    #f_range=np.linspace(start_freq*1e9, stop_freq*1e9, points)
    f_range=np.linspace(start_freq, stop_freq, points)
    
    
    #   (2pi* L * i*omega) + 1
    #   ----------------------
    #      2pi * C * i*omega 
    #LC_circuit=(1j*L*2*np.pi*f_range+1/(1j*C*2*np.pi*f_range))#frequency dependant impedance
    LCR_circuit=(1j*L*2*np.pi*f_range+1/(1j*C*2*np.pi*f_range))+R*np.ones(len(f_range))

    
    cavity_sparams=create_sparams_from_ABCD(1,LCR_circuit,0,1,Z0)
    eps = 1e-32
    cavity_sparams = np.where((cavity_sparams==0), eps,cavity_sparams)
    
    path= Path('./data/touchstone_files/cavities/')
    file_name = 'LC_cavity'
    file_name_ext = file_name + '.s2p'
    expected_full_name_and_path = path / file_name_ext
        
    #if save: create_s2p(f_range,cavity_sparams,path,file_name)
    create_s2p(f_range,cavity_sparams,path,file_name)
    #cavity_network=rf.Network(expected_full_name_and_path,f_unit='GHz')
    cavity_network=rf.Network(str(expected_full_name_and_path))

    return(cavity_network)




    
    
    



def make_antenna_network(N,start_freq, stop_freq, points = 10001): # , save = True
   
    
    #pts=20001
    #f_range=np.linspace(start_freq*1e9, stop_freq*1e9, points)
    f_range=np.linspace(start_freq, stop_freq, points)

    N = N * 1.0 # make sure it's not an int  
    A = np.multiply(np.ones_like(f_range),N)
    B = 0
    C = 0
    D = np.multiply(np.ones_like(f_range),(1/N))
    Z0 = 50
    
    ant_sparams=create_sparams_from_ABCD(A,B,C,D,Z0)
    eps = 1e-32
    ant_sparams = np.where((ant_sparams==0), eps,ant_sparams)

    
    path= Path('./data/touchstone_files/antennas/')
    #file_name = 'antenna_N='+str(N)
    file_name = 'antenna'
    file_name_ext = file_name + '.s2p'
    expected_full_name_and_path = path / file_name_ext 
        
    #if save: create_s2p(f_range,ant_sparams,path,file_name)
    create_s2p(f_range,ant_sparams,path,file_name)
    antennna_network =rf.Network(str(expected_full_name_and_path),f_unit='GHz')

    return(antennna_network)



def calc_C(f_res,L_in):
    return_c=1/(((2*np.pi*f_res)**2)*L_in)
    return(return_c)
    
def calc_L(f_res,C_in):
    return_l=1/(((2*np.pi*f_res)**2)*C_in)
    return(return_l)



def calc_L_and_C(f_res,R_in,Q_in):
    return_l=Q_in*R_in/(2*np.pi*f_res)
    return_c=1/(2*np.pi*f_res*Q_in*R_in)
    return(return_l,return_c)



def import_rf_data(data_path,fname,start_f,stop_f,f_pts):
    #path='\\\pnl\\projects\\admx\\People\\Tedeschi\\2-4GHz_electronic_tuning\\s2p_data\\'
    new_frange=rf.Frequency(f_start,f_stop,f_pts,'Hz')
    
    data_raw=rf.Network(data_path + fname, f_unit='Hz')
    
    data_raw.crop(start_f,stop_f)

    data_raw.frequency=rf.Frequency(start_f,stop_f,len(data_raw.frequency),unit='Hz')#explicitly setting the frequency range, not this can cause slight frequency errors in coarsely sampled data
    
    data=data_raw.interpolate(new_frange,kind='cubic')
    
    return(data)

def mils_to_meters(mils):
    meters=(mils*2.54)/(1000*100)
    return(meters)
    


########################### creating s param from ABCDZ ###########################

def create_sparams_from_ABCD(A,B,C,D,Z0):
    #equations adopted from Pozar
    S11=(A + B/Z0 - C*Z0 - D)/(A + B/Z0 - C*Z0 + D)
    
    S12=2*(A*D-B*C)/(A+B/Z0+C*Z0+D)
    
    S21=2/(A+B/Z0+C*Z0+D)
    
    S22=(-1*A + B/Z0 - C*Z0 + D)/(A + B/Z0 - C*Z0 + D)
    
    return(np.flipud(np.rot90(np.array([S11,S12,S21,S22]))))


    
    
    
    
    
    
    
    
    
########################### create sNp file ###########################

##### s2p
def create_s2p(frequency_range,complex_data_matrix,s2p_path,name,*fmt):
    comment = '# Hz S  lin   R 50'
    
    s11_mag=(np.abs(complex_data_matrix[:,0]))
    s11_phase=np.angle(complex_data_matrix[:,0])*(180/np.pi)
    
    s21_mag=(np.abs(complex_data_matrix[:,2])) #touchstone format = s11,s21,s12,s22
    s21_phase=np.angle(complex_data_matrix[:,2])*(180/np.pi)
    
    s12_mag=(np.abs(complex_data_matrix[:,1]))
    s12_phase=np.angle(complex_data_matrix[:,1])*(180/np.pi)
    
    s22_mag=(np.abs(complex_data_matrix[:,3]))
    s22_phase=np.angle(complex_data_matrix[:,3])*(180/np.pi)
    
    s2p_matrix=np.flipud(np.rot90(np.array([frequency_range/1e9,s11_mag,s11_phase,s21_mag,s21_phase,s12_mag,s12_phase,s22_mag,s22_phase])))
    
    name_ext = name+'.s2p'
    s2p_path_name_ext = s2p_path / name_ext

    np.savetxt(s2p_path_name_ext,s2p_matrix,delimiter='  ',header='!S2P File: Measurements: S11, S21, S12, S22:/n' + comment)





##### s3p
def create_s3p(frequency_range,complex_data_matrix,s2p_path,name,*fmt):
    comment = '# Hz S  lin   R 50'
    
    s11_mag=(np.abs(complex_data_matrix[:,0]))
    s11_phase=np.angle(complex_data_matrix[:,0])*(180/np.pi)
    
    s12_mag=(np.abs(complex_data_matrix[:,1]))
    s12_phase=np.angle(complex_data_matrix[:,1])*(180/np.pi)
    
    s13_mag=(np.abs(complex_data_matrix[:,2]))
    s13_phase=np.angle(complex_data_matrix[:,2])*(180/np.pi)   
        
    s21_mag=(np.abs(complex_data_matrix[:,3])) #touchstone format = s11,s21,s12,s22
    s21_phase=np.angle(complex_data_matrix[:,3])*(180/np.pi)
       
    s22_mag=(np.abs(complex_data_matrix[:,4]))
    s22_phase=np.angle(complex_data_matrix[:,4])*(180/np.pi)

    s23_mag=(np.abs(complex_data_matrix[:,5]))
    s23_phase=np.angle(complex_data_matrix[:,5])*(180/np.pi)

    s31_mag=(np.abs(complex_data_matrix[:,6])) #touchstone format = s11,s21,s12,s22
    s31_phase=np.angle(complex_data_matrix[:,6])*(180/np.pi)
       
    s32_mag=(np.abs(complex_data_matrix[:,7]))
    s32_phase=np.angle(complex_data_matrix[:,7])*(180/np.pi)

    s33_mag=(np.abs(complex_data_matrix[:,8]))
    s33_phase=np.angle(complex_data_matrix[:,8])*(180/np.pi)
    
    s2p_matrix=np.flipud(np.rot90(np.array([frequency_range/1e9,s11_mag,s11_phase,s12_mag,s12_phase,s13_mag,s13_phase,s21_mag,s21_phase
                                            ,s22_mag,s22_phase,s23_mag,s23_phase,s31_mag,s31_phase,s32_mag,s32_phase,s33_mag,s33_phase])))
    
    np.savetxt(s2p_path+name+'.s2p',s2p_matrix,delimiter='  ',header='!S3P File: Measurements: S11, S12, S13, S21, S22, S23, S31, S32, S33:/n' + comment)




def break_apart_2port_network(network_2port,*magnitude_only):
    net_s11=network_2port.s[:,0,0]
    net_s12=network_2port.s[:,0,1]
    net_s21=network_2port.s[:,1,0]
    net_s22=network_2port.s[:,1,1]
    
    return(net_s11,net_s12,net_s21,net_s22)

########################### interpolate ###########################


def linear_interpolate(array,pts):
    interp_base=np.linspace(1,10,len(array))
    interp_newbase=np.linspace(1,10,pts)
    interpolated_array=np.interp(interp_newbase,interp_base,array)
    return(interpolated_array)

def spline_interpolate(x_values,y_values,order,pts):
    #Assumes x_values is linear    
    new_x_values=linear_interpolate(x_values,pts)   
    interpolated_array=sc_interp.spline(x_values,y_values,new_x_values,order,kind='smoothest',conds=None)    
    return(interpolated_array)
    
def cubic_interpolate(x_values,y_values,pts):
    new_x_values=linear_interpolate(x_values,pts)   
    cubic_function=sc_interp.interp1d(x_values,y_values,kind='cubic')#creates a cubic function that can be evaulated within the range
    new_y_values=cubic_function(new_x_values)
    return(new_x_values,new_y_values)

################### Calculate Q from S params (Jonathan) ###########################


def calc_Q_s21(frequencies_forQ,s21_forQ,dB_from_peak,*designed_Q):

    f_interp=linear_interpolate(frequencies_forQ,1000000)
    s21_interp=linear_interpolate(s21_forQ,1000000)
    
    dB_bw=dB_from_peak-np.max(s21_interp)
    
    # =============================================================================
    # Loaded Q Calculations   
    # =============================================================================
    # -3dB BW from 0dB
    start_index=np.argmin(np.abs(s21_interp[:np.argmax(s21_interp)]+dB_bw))
    print(start_index)
    stop_index=np.argmin(np.abs(s21_interp[np.argmax(s21_interp):]+dB_bw))+np.argmax(s21_interp)
    print(stop_index)
    loaded_Q = ((f_interp[stop_index]-f_interp[start_index])/2+f_interp[start_index])/(f_interp[stop_index]-f_interp[start_index])

    # =============================================================================
    # Unloaded Q Calculations   
    # =============================================================================
    unloaded_Q=loaded_Q/(1-10**(np.max(s21_interp)/20))
    
    plt.figure()
    plt.plot(f_interp,s21_interp)
    plt.scatter([f_interp[start_index],f_interp[stop_index]],[s21_interp[start_index],s21_interp[stop_index]],color='r',label='loaded Q')
    plt.legend()
    plt.xlim(np.floor(f_interp[start_index]/1e9)*1e9-.1e9,np.ceil(f_interp[start_index]/1e9)*1e9+.1e9)
    plt.ylabel('dB')
    plt.xlabel('Hz')
    plt.grid()
    if bool(designed_Q):
        plt.title('S21 Calculated Cavity Unloaded Q:'+str(np.round(unloaded_Q,decimals=2))+
                  '\nS21 Calculated Cavity Loaded Q:'+str(np.round(loaded_Q,decimals=2))+
                  '\nDesigned RLC Q: '+str(designed_Q[0]))
    else:
        plt.title('S21 Calculated Cavity Unloaded Q:'+str(np.round(unloaded_Q,decimals=2))+
                  '\nS21 Calculated Cavity Loaded Q:'+str(np.round(loaded_Q,decimals=2)))
    
    

def calc_Q_s11(frequencies_forQ,s11_forQ,dB_bw,*designed_Q):
    #NOTE, unloaded Q based on S11 has accuracy issues if the data set is not highly sampled
    
    #dB_bw assumed to be negative, eg -3dB BW
    
    f_interp=linear_interpolate(frequencies_forQ,1000000)
    s11_interp=linear_interpolate(s11_forQ,1000000)
   

    # =============================================================================
    # Loaded Q Calculations   
    # =============================================================================
    # -3dB BW from 0dB
    start_index=np.argmin(np.abs(s11_interp[:np.argmin(s11_interp)]+dB_bw-np.max(s11_interp)))
    print(start_index)
    stop_index=np.argmin(np.abs(s11_interp[np.argmin(s11_interp):]+dB_bw-np.max(s11_interp)))+np.argmin(s11_interp)
    print(stop_index)
    loaded_Q = ((f_interp[stop_index]-f_interp[start_index])/2+f_interp[start_index])/(f_interp[stop_index]-f_interp[start_index])

    # =============================================================================
    # Unloaded Q Calculations   
    # =============================================================================
    #  +3dB from min
    UL_start_index=np.argmin(np.abs(s11_interp[:np.argmin(s11_interp)]-(dB_bw+s11_interp[np.argmin(s11_interp)])))
    print(start_index)
    UL_stop_index=np.argmin(np.abs(s11_interp[np.argmin(s11_interp):]-(dB_bw+s11_interp[np.argmin(s11_interp)])))+np.argmin(s11_interp)
    print(stop_index)   
    unloaded_Q = ((f_interp[UL_stop_index]-f_interp[UL_start_index])/2+f_interp[UL_start_index])/(f_interp[UL_stop_index]-f_interp[UL_start_index])

    
    plt.figure()
    plt.plot(f_interp,s11_interp)
    plt.scatter([f_interp[start_index],f_interp[stop_index]],[s11_interp[start_index],s11_interp[stop_index]],color='r')
    plt.scatter([f_interp[UL_start_index],f_interp[UL_stop_index]],[s11_interp[UL_start_index],s11_interp[UL_stop_index]],color='K')

    plt.xlim(np.floor(f_interp[start_index]/1e9)*1e9-.1e9,np.ceil(f_interp[start_index]/1e9)*1e9+.1e9)
    plt.ylabel('dB')
    plt.xlabel('Hz')
    plt.grid()
    if bool(designed_Q):
        plt.title('S11 Calculated Cavity Unloaded Q:'+str(np.round(unloaded_Q,decimals=2))+
                  '\nS11 Calculated Cavity Loaded Q:'+str(np.round(loaded_Q,decimals=2))+
                  '\nDesigned RLC Q: '+str(designed_Q[0]))
    else:
        plt.title('S11 Calculated Cavity Unloaded Q:'+str(np.round(unloaded_Q,decimals=2))+
                  '\nS11 Calculated Cavity Loaded Q:'+str(np.round(loaded_Q,decimals=2)))

       
    
    return(unloaded_Q,loaded_Q)


######################################################################################################################
## Function to make passive devices into noisy networks
######################################################################################################################
## From SNP files
def read_file_in_noisy_network(file_path,file_name,freq_interp,Tphys):
    file_loc = str(Path(file_path) / file_name)
    file_net = rf.Network(file_loc)
    file_net_interp = file_net.interpolate(freq_interp, kind='cubic')
    file_net_interp_noisy = rf.NoisyNetwork(file_net_interp)
    file_net_interp_noisy.noise_source(source='passive',T0 = Tphys)

    return file_net_interp_noisy

## From 'Network' of SNP files
def read_in_noisy_network(file_net,freq_interp,Tphys):
    file_net_interp = file_net.interpolate(freq_interp, kind='cubic')
    file_net_interp_noisy = rf.NoisyNetwork(file_net_interp)
    file_net_interp_noisy.noise_source(source='passive',T0 = Tphys)

    return file_net_interp_noisy

###################################################################################################################
# Function to return linear NF from covariance matrix
###################################################################################################################
def return_nf(port, noise_ntwk):
    #Currently must be a two port device

    I = np.identity(np.shape(noise_ntwk.s)[1])
    I[port-1,port-1] = np.zeros_like(noise_ntwk.cs[0,port-1,port-1],dtype=float)
    AS = np.matmul(I, np.conjugate(noise_ntwk.s.swapaxes(1, 2)))
    SAS = np.matmul(noise_ntwk.s, AS)
    F2 = 1 +  np.real(noise_ntwk.cs[:,port-1,port-1]) / (K_BOLTZMANN*290*SAS[:,port-1,port-1])

    return F2
