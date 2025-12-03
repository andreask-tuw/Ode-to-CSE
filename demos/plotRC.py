import numpy as np

data = np.loadtxt('output_test_ode.txt', usecols=(0, 1))

import matplotlib.pyplot as plt

plt.plot(data[:,0], data[:,1], label='voltage')
# plt.plot(data[:,0], data[:,2], label='velocity')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Resistor-Capacitor System Time Evolution')
plt.legend()
plt.grid()
plt.show()

