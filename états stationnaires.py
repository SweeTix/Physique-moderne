import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Paramètres physiques
L = 1.0               # Longueur du domaine (en unités arbitraires)
N = 1000              # Nombre de points de discrétisation
dx = L / N            # Pas d'espace
x = np.linspace(0, L, N)

# Potentiel : puits de profondeur V0 < 0 entre x1 et x2
V0 = -4000.0
x1, x2 = 0.4 * L, 0.6 * L
V = np.zeros(N)
V[(x > x1) & (x < x2)] = V0

# Construction du Hamiltonien tridiagonal : méthode des différences finies
main_diag = (1.0 / dx**2) * np.ones(N) + V
off_diag = (-0.5 / dx**2) * np.ones(N - 1)

# Résolution du problème aux valeurs propres
eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

# Sélection des états liés (E < 0)
bound_indices = np.where(eigenvalues < 0)[0]
n_bound = len(bound_indices)
print(f"Nombre d'états liés trouvés : {n_bound}")

# Affichage des premiers états liés
plt.figure(figsize=(10, 6))
plt.plot(x, V / np.abs(V0), 'k--', label="Potentiel (échelle relative)")

for i in range(n_bound):
    psi = eigenvectors[:, bound_indices[i]]
    psi /= np.sqrt(np.sum(psi**2) * dx)  # normalisation
    plt.plot(x, psi + eigenvalues[bound_indices[i]] / np.abs(V0), label=f"ψ{i}, E={eigenvalues[bound_indices[i]]:.2f}")

plt.title("États stationnaires dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("Ψ(x) + E (relatif)")
plt.legend()
plt.grid()
plt.show()