import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

def plot_txt(filename):
    data = np.loadtxt(filename)
    print(data)

plot_txt('spectre_25pers.txt')