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
    Calculate the system energy, without optimization, for all the frequencies of the plage.
    
    :param plage: numpy.array([[wavenumber, inciding_wave_power/energy], [], ...], type=float)
    :return energies: numpy.array([[wavenumber, system_energy], [], ...], type=float)
    '''
    energies = []
    for i in range(len(plage)):
        energy = demo_control_polycopie2024.launch_experience(wavenumber=plage[i][0], incident_wave_energy=plage[i][1], optimize=False)
        energies.append([energy,plage[i][0]])
    print(energies)
    energies = numpy.array(energies)
    print(energies)
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
    plage_test_1 = numpy.array([[10**(k), 1] for k in range(-3,4)])
    energies = get_energies_frequency_plage(plage_test_1)
    plot_energy_distribution(energies)