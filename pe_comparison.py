# step 1: generate some white noise from A+, CE etc
# step 2: inject a NR waveform
# step 2: perform PE using 3 damped sinusoid (P. Easter's model)
# step 4: perform PE using tBilby 
# step 5: Compare PE results
# step 6: benchmark for different signal durations
import os
CWD = os.getcwd()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey

import pmtoolkit as pmt
import bilby
import tbilby
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import ConditionalLogUniform, LogUniform,Uniform


print('starting script...')
# Set up some signals processing stuff
sample_frequency = int(8192) # must be an int
signal_duration = 2*sample_frequency # can be a float
master_time = np.arange(0, signal_duration, 1/sample_frequency)
start_time = 0.0

######################
#     Injection      #
######################


NR_waveform_arguments = dict(
    t_0 = 0.0,
)

# define parameters to inject.
injection_parameters = dict(    
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    t_0=0,
    geocent_time=0.0,
)

# load and use a NR wavefrom
NR_waveform_master = pmt.NRWaveform("THC_0036-master-R03/R03", loudness = 1E-20)

# # extract this waveform to use it with Bilby/tBilby:
def NRmodel(time, **kwargs):
    ret = NR_waveform_master.interpolate_wf_to_new_tarray(time, kwargs['t_0'], window = True, tukey_rolloff = 0.2)
    return ret

# load the NR waveform into Bilby
NR_waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=signal_duration,
    sampling_frequency=sample_frequency,
    time_domain_source_model=NRmodel,
    waveform_arguments=NR_waveform_arguments,
    start_time=injection_parameters["geocent_time"],
)

# Use Bilby to generate some Gaussian noise, and inject the signal - H1 L1 V1 for now
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

# need to manually set the upper frequency limit as it defaults to 2048
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
    interferometer.maximum_frequency = sample_frequency/2

## Generate Gaussian noise
interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=sample_frequency,
        duration=signal_duration, start_time=start_time)

## Generate zero noise 
#interferometers.set_strain_data_from_zero_noise(sampling_frequency=sample_frequency, duration=signal_duration)


## Inject signal into noise
interferometers.inject_signal(parameters=injection_parameters, 
        waveform_generator=NR_waveform, raise_error = False)

# plot these to check its working
outdir='outdir'
label='noise'
# interferometers.plot_data(outdir=outdir, label=label)

####################
#       Model      #
####################

def damped_sin(time, t_0, amplitude, damping_time, frequency, phase, drift):

    """
    damping time in ms
    frequency in Hz
    amplitude in log10
    :amplitude, damping_time, frequency, phase, drift):
    """
    # sample_rate = int(8192)#time[1] - time[0]
    # duration = 2*sample_frequency#max(time)


    # ret = pmt.damped_sinusoid_td(sample_rate, duration, amplitude = amplitude,
    #                                                 damping_time = damping_time,
    #                                                 frequency = frequency,
    #                                                 phase = phase,
    #                                                 drift = drift,
    #
    #                                                 weight = None)

    time_inx = time.copy() >= t_0
    t = time.copy()[time_inx] - t_0 # start from zero offset 

    hplus = np.zeros(len(time))
    hcross = hplus.copy()

    if drift is not None:
        h_cplx = amplitude*np.exp(-t/damping_time)*\
                                    np.exp(1j * ( 2*np.pi*frequency*t*(1+drift*t) + phase))

        hplus += np.imag(h_cplx)
        hcross += np.real(h_cplx)
            # check of nan or overflow 
        if any(np.isnan(h_cplx)):
            print('this is dead...')
    else:
        A = amplitude*np.exp(-t/damping_time)
        theta = 2*np.pi*frequency*t + phase

        hplus += A * np.sin(theta)
        hcross += A * np.cos(theta)

    # indtroducing a window       
    window_dt = t[-1]-t[0]
    tukey_rolloff_ms=  0.2/1000    
    window = tukey(len(t), 2 * tukey_rolloff_ms / window_dt)

    hplus[time_inx] =  hplus * window
    hcross[time_inx] =  hcross * window

    return {'plus': hplus, 'cross': hcross}


n_comp = int(3)
component_functions_dict = {}
component_functions_dict[damped_sin] = (n_comp, 'amplitude', 
                                                'damping_time', 'frequency',
                                                 'phase', 'drift')

model = tbilby.core.base.create_transdimensional_model('model',  component_functions_dict,
        returns_polarization=True,
        SaveTofile=True)


############################
#          priors          #
############################

# require to enforce ordering of frequencies/amplitudes

# class TransdimensionalConditionalUniform_f(tbilby.core.prior.TransdimensionalConditionalUniform):   
#     def transdimensional_condition_function(self,**required_variables):
#         # setting the mimmum according the the last peak value
#             minimum = self.minimum
#             if(len(self.f_Hz)>0): # handle the first mu case
#                 minimum = self.f_Hz[-1] # set the minimum to be the location of the last peak 
                           
#             return dict(minimum=minimum)


# class TransdimensionalConditionalLogUniform_A(tbilby.core.prior.TransdimensionalConditionalLogUniform):
#     def transdimensional_condition_function(self,**required_variables):
#         # setting the mimmum according the the last peak value
#             minimum = self.minimum
#             if(len(self.A)>0): # handle the first mu case
#                 minimum = self.A[-1] # set the minimum to be the location of the last peak

#             return dict(minimum=minimum)


priors_t = bilby.core.prior.dict.ConditionalPriorDict()
# Number of damped sinusoid components
priors_t ['n_damped_sin'] = tbilby.core.prior.DiscreteUniform(1,n_comp,'n_damped_sin')
# Amplitude, plain priors
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'amplitude',n_comp,
                                prior_dict_to_add=priors_t,minimum=-24, maximum=-19)
# Frequency
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'frequency',n_comp,
                                prior_dict_to_add=priors_t,minimum=1000, maximum=5000)
priors_t  = tbilby.core.base.create_plain_priors(LogUniform,'damping_time',n_comp,
                            prior_dict_to_add=priors_t,minimum=0.5, maximum=50)
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'phase',n_comp,
                            prior_dict_to_add=priors_t,minimum=-np.pi, maximum=np.pi)
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'drift',n_comp,
                            prior_dict_to_add=priors_t,minimum=-6.4 , maximum=6.4 )

priors_t['t_0']  = bilby.prior.Uniform(minimum=-1.5/1000,maximum =1.5/1000)

sample_for_injection = priors_t.sample(5)

constant_priors = injection_parameters.copy()
for k in constant_priors.keys():
    priors_t[k]=constant_priors[k]


#################
#  LIKELIHOOD    #
#################

waveform_arguments={}
waveform_arguments['t_0']=0.0

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=signal_duration,
    sampling_frequency=sample_frequency,
    time_domain_source_model=model,
    start_time=waveform_arguments['t_0'],
    
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers,waveform)

#######################
#     Run Sampler     #
#######################
print('starting sampler...')
result = bilby.core.sampler.run_sampler(
likelihood,
priors=priors_t,
injection_parameters=injection_parameters,
sampler='dynesty',
label='noise',
clean=True,
nlive=10,
outdir='outdir', 
verbose = True  
    )


















