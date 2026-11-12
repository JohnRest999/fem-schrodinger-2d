"""Genera la malla externa de ejemplo (cuadrado de 100 nm de lado, h = 0.8 nm) en
el formato de texto que lee caso.py: nodos (x y), elementos (i j k), aristas (i j etiqueta)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scipy.spatial import Delaunay
from pruebas import _interior_hex
a, h = 50.0, 0.8
n = int(round(2 * a / h))
s = np.linspace(-a, a, n + 1)
borde = np.unique(np.vstack([np.c_[s, -a + 0 * s], np.c_[s, a + 0 * s], np.c_[-a + 0 * s, s], np.c_[a + 0 * s, s]]), axis=0)
g = _interior_hex(-a, a, -a, a, h); g = g[(np.abs(g) < a - 0.6 * h).all(1)]
P = np.vstack([borde, g]); T = Delaunay(P).simplices
ar = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
u, c = np.unique(ar, axis=0, return_counts=True); fr = u[c == 1]
d = os.path.join(os.path.dirname(__file__), "malla_cuadrada")
np.savetxt(os.path.join(d, "nodos.txt"), P, fmt="%.10f")
np.savetxt(os.path.join(d, "elementos.txt"), T, fmt="%d")
with open(os.path.join(d, "aristas.txt"), "w") as f:
    for i, j in fr: f.write(f"{i} {j} borde\n")
print(len(P), "nodos,", len(T), "elementos,", len(fr), "aristas de frontera")
