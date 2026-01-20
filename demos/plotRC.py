import numpy as np
import matplotlib.pyplot as plt

methods = ['ExplicitEuler', 'ImprovedEuler', 'ImplicitEuler', 'CrankNicholson', 'IRK', 'ERK']

filenames = [f'../build/output_test_ode_RC_{method}.txt'
            for method
            in methods]

for idx, filename in enumerate(filenames):
    data = np.loadtxt(filename, usecols=(0, 1))

    plt.plot(data[:,0], data[:,1], label='voltage')
    # plt.plot(data[:,0], data[:,2], label='velocity')
    plt.xlabel('time')
    plt.ylabel('voltage')
    plt.title('Resistor-Capacitor System Time Evolution\n' + methods[idx])
    # plt.legend()
    plt.grid()
    plt.show()

