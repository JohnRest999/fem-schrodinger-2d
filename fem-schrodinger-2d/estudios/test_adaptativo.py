"""Pruebas de la Fase 4: conformidad y calidad de NVB, y comportamiento del estimador."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import numpy as np
from pruebas import malla_heteroestructura
from adaptativo import aristas, refinar_nvb, calidad, marcar_dorfler, estimador
from fem_schrodinger2d import FEMSchrodinger2D, geometria

def conforme(m):
    E2, ed, vec = aristas(m)
    frontera = (vec[:, 1] < 0).sum()
    A, _, _ = geometria(m)
    return frontera == len(m.aristas), (A > 0).all(), A.sum()

rng = np.random.default_rng(1)
m = malla_heteroestructura(2.0, 0.25, [(1.0, (0, 0))], ajustada=True)
print(f"inicial: N={m.n}, ángulo mín={calidad(m):.1f}°, área={geometria(m)[0].sum():.6f} (pi*4={np.pi*4:.6f})")
for it in range(12):
    marc = rng.choice(len(m.elems), size=max(1, len(m.elems) // 10), replace=False)
    m = refinar_nvb(m, marc)
    ok_fr, ok_A, area = conforme(m)
    print(f"it {it:2d}: N={m.n:6d} elems={len(m.elems):6d} frontera_ok={ok_fr} areas>0={ok_A} "
          f"ángulo mín={calidad(m):5.1f}° área={area:.6f}")
# nodos de frontera e interfaz siguen sobre sus círculos
r = np.hypot(*m.nodos[m.nodos_de('borde')].T)
print("máx |r-2| en frontera:", np.abs(r - 2).max())
