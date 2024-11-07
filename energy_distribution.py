# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import postprocessing
import demo_control_polycopie2024

def get_sound_tested():
    '''
    
    :param : un fichier texte de type spectre, comme dans sons_et_spectres
    :return: numpy.array([[wavenumber, inciding_wave_power/energy], [], ...], type=float)
    '''
    pass

def get_energies_frequency_plage(plage):
    '''
    Calculate the system energy, without optimization, for all the wavenumbers of the plage.
    
    :param plage: numpy.array([[wavenumber, inciding_wave_power/energy], [], ...], type=float)
    :return energies: numpy.array([[wavenumber, system_energy], [], ...], type=float)
    '''
    energies = []
    c = 343 # speed of sound in the field (air for now)
    for i in range(len(plage)):
        print(i)
        frequency = plage[i][0]
        wavenumber = ( 2 * numpy.pi / c ) * frequency
        energy = demo_control_polycopie2024.launch_experience(wavenumber=wavenumber, incident_wave_energy=plage[i][1], optimize=False)
        energies.append([energy,plage[i][0]])
        
    energies = numpy.array(energies)
    
    return energies

def plot_energy_distribution(energies):
    '''
    Plot the energy against the wavenumber
    
    :param energies: numpy.array([[wavenumber, system_energy], [], ...], type=float)
    '''
    frequencies = [energies[k][1] for k in range(len(energies))]
    energies_only = [energies[k][0] for k in range(len(energies))]

    matplotlib.pyplot.plot(frequencies, energies_only, marker='o', linestyle='-', color='b')  
    matplotlib.pyplot.xlabel("Frequencies")  
    matplotlib.pyplot.ylabel("Energies")   
    matplotlib.pyplot.title('Energy distribution') 
    
    matplotlib.pyplot.savefig("fig_energy_distribution.png", format='png', dpi=300)  # dpi=300 for high resolution
    matplotlib.pyplot.close()
    
if __name__ == '__main__':
    plage_frequencies_test = numpy.arange(20,1700,5)
    plage_test_1 = []
    for i in range(len(plage_frequencies_test)):
        plage_test_1.append([plage_frequencies_test[i], 1])
    plage_test_1 = numpy.array(plage_test_1)
    print(plage_test_1)
    
    energies = get_energies_frequency_plage(plage_test_1)
    plot_energy_distribution(energies)