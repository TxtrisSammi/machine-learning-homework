from matplotlib import pyplot as plt
import numpy as np

"""
Simple parabola without labels and captions.
"""

plt.plot([-3, -2, -1, 0, 1, 2, 3], [9, 4, 1, 0, 1, 4, 9])
plt.show()

"""
Proper figures with labels and captions.
"""

# x = np.arange(-3, 4)
x = np.linspace(-np.pi, np.pi, 10)
y = x**2
z = np.sin(x)

plt.title('Parabola & Sinusoidal Curves')
plt.plot(x, y, 'ko', label=r'$x^2$')
plt.plot(x, z, 'cx-', label=r'$\sin(x)$')
plt.xlabel('Domain')
plt.ylabel('Range')
plt.tight_layout()
plt.show()

"""
Plotting images or matrices
"""

# The magic AI one liner
# X = (np.arange(64).reshape(8, 8) % 2) ^ (np.arange(8)[:, None] % 2 == 0)

X = np.arange(8*8).reshape(8, 8) % 2
for i in range(0, X.shape[0], 2):
    X[i] = (X[i] + 1) % 2

# The understandable AI one liner
# X[1::2] = (X[::2] + 1) % 2

plt.imshow(X, cmap='grey')
plt.tight_layout()
plt.show()


"""
Euler's Number Series for n in Naturals
"""

n = 7
i = np.arange(n)
i[1:] = np.cumprod(i[1:])
i[0] = 1
i = 1 / i
e = np.cumsum(i)
error = (np.e - e)**2

plt.title('Convergence to e.')
plt.plot(error)
plt.xlabel('n')
plt.ylabel(r'$\left(e - \sum_{i=0}^{n}\frac{1}{i!}\right)^2$')
plt.tight_layout()
plt.show()
