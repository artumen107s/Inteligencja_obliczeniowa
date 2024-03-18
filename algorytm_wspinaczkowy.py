import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def michal(xx, m=10):
    ii = np.arange(1, len(xx) + 1)
    sin_part = np.sin(xx) * (np.sin(ii * xx**2 / np.pi))**(2 * m)
    summation = np.sum(sin_part)
    y = -summation
    return y

def hill_climbing_michal(max_iter=100, step_size=0.05):
    path = []
    current_point = np.array([random.uniform(0, np.pi), random.uniform(0, np.pi)])
    current_value = michal(current_point)
    path.append(current_point)

    for _ in range(max_iter):
        new_point = current_point + np.array([random.uniform(-step_size, step_size), random.uniform(-step_size, step_size)])
        new_value = michal(new_point)

        if new_value < current_value:
            current_point = new_point
            current_value = new_value
        
        path.append(current_point)

    return np.array(path)

# Generowanie ścieżki algorytmu wspinaczkowego
path = hill_climbing_michal()

# Przygotowanie danych do wykresu
x1_values = np.linspace(0, np.pi, 100)
x2_values = np.linspace(0, np.pi, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = np.zeros_like(X1)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = michal(np.array([X1[i, j], X2[i, j]]))

# Wykres 3D funkcji Michalewicza
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)

# Dodanie ścieżki algorytmu wspinaczkowego
path_z = np.array([michal(p) for p in path])
ax.plot3D(path[:, 0], path[:, 1], path_z, 'r-', linewidth=2, marker='o', markersize=4, markerfacecolor='black')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('michal([X1, X2])')
ax.set_title('Michalewicz Function')

plt.show()
