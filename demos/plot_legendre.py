import matplotlib.pyplot as plt
import numpy as np

try:
    data = np.loadtxt("legendre.txt")
except OSError:
    print("Error: legendre.txt not found. Run demo_autodiff first.")
    exit(1)

x = data[:, 0]
P = data[:, 1:7]
dP = data[:, 7:13]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i in range(6):
    plt.plot(x, P[:, i], label=f'P_{i}')
plt.title("Legendre Polynomials")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
for i in range(6):
    plt.plot(x, dP[:, i], label=f"P'_{i}")
plt.title("Derivatives of Legendre Polynomials")
plt.legend()
plt.grid()

plt.savefig("legendre_plot.png")
plt.show()
