import numpy as np
import matplotlib.pyplot as plt

methods = ['ExplicitEuler', 'ImprovedEuler', 'ImplicitEuler', 'CrankNicholson', 'IRK', 'ERK']

filenames = [f'../build/output_test_ode_{method}.txt'
            for method
            in methods]

for idx, filename in enumerate(filenames):

    data = np.loadtxt(filename, usecols=(0, 1, 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(data[:,0], data[:,1], label='position')
    ax1.plot(data[:,0], data[:,2], label='velocity')
    ax1.set_xlabel('time')
    ax1.set_ylabel('value')
    ax1.set_title('Mass-Spring System Time Evolution\n' + methods[idx])
    ax1.legend()
    ax1.grid()
    
    ax2.plot(data[:,1], data[:,2], label='phase plot')
    ax2.set_xlabel('position')
    ax2.set_ylabel('velocity')
    ax2.set_title('Mass-Spring System Phase Plot\n' + methods[idx])
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()
