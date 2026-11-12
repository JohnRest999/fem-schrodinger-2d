"""
pruebas.py
==========
Generación de mallas, gráficos y pruebas de validación del solver
(fem_schrodinger2d.py). Aquí se definen las constantes físicas, el potencial
y las condiciones de frontera, y se le pasan al solver.
"""
import numpy as np
import sympy as sp
from scipy.spatial import Delaunay

from fem_schrodinger2d import Malla, FEMSchrodinger2D
from fem_schrodinger2d import x as sx, y as sy

# ----------------------------------------------------------------------
# Constantes físicas (SI) y conversión a eV*nm^2
# ----------------------------------------------------------------------
HBAR = 1.054571817e-34        # J s
M_E = 9.1093837015e-31        # kg
Q_E = 1.602176634e-19         # C
M_GAAS = 0.067 * M_E          # masa efectiva GaAs
J_M2_A_EV_NM2 = 1.0 / (Q_E * 1e-18)   # J*m^2 -> eV*nm^2


# ======================================================================
# MALLAS
# ======================================================================
def _delaunay_dominio(pts, dentro, clasificar):
    tri = Delaunay(pts)
    c = pts[tri.simplices].mean(axis=1)
    ok = dentro(c[:, 0], c[:, 1])
    elems = tri.simplices[ok]
    usados = np.unique(elems)
    mapa = -np.ones(len(pts), int)
    mapa[usados] = np.arange(len(usados))
    return Malla.desde_triangulos(pts[usados], mapa[elems], clasificar)


def _circulo(R, h, c=(0.0, 0.0)):
    n = max(8, int(round(2 * np.pi * R / h)))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + R * np.cos(t), c[1] + R * np.sin(t)]


def _interior_hex(xmin, xmax, ymin, ymax, h):
    pts = []
    dy = h * np.sqrt(3) / 2
    for j, y in enumerate(np.arange(ymin, ymax + dy, dy)):
        off = 0.5 * h if j % 2 else 0.0
        pts += [(x, y) for x in np.arange(xmin + off, xmax + h, h)]
    return np.array(pts)


def malla_disco(R, h):
    """Disco de radio R, tamaño de malla ~h. Etiqueta de frontera: 'borde'."""
    frontera = _circulo(R, h)
    g = _interior_hex(-R, R, -R, R, h)
    g = g[np.hypot(g[:, 0], g[:, 1]) < R - 0.6 * h]
    pts = np.vstack([frontera, g])
    dentro = lambda x, y: np.hypot(x, y) < R
    return _delaunay_dominio(pts, dentro, lambda m: "borde")


def malla_anillo_asimetrico(R_ext, R_int, x0, y0, h):
    """Anillo con agujero excéntrico. Etiquetas: 'exterior' e 'interior'."""
    if np.hypot(x0, y0) + R_int >= R_ext:
        raise ValueError("El agujero no está contenido en el disco exterior.")
    ext = _circulo(R_ext, h)
    inn = _circulo(R_int, h, (x0, y0))
    g = _interior_hex(-R_ext, R_ext, -R_ext, R_ext, h)
    d_ext = np.hypot(g[:, 0], g[:, 1])
    d_int = np.hypot(g[:, 0] - x0, g[:, 1] - y0)
    g = g[(d_ext < R_ext - 0.6 * h) & (d_int > R_int + 0.6 * h)]
    pts = np.vstack([ext, inn, g])

    def dentro(x, y):
        return (np.hypot(x, y) < R_ext) & (np.hypot(x - x0, y - y0) > R_int)

    def clasificar(m):
        de = abs(np.hypot(*m) - R_ext)
        di = abs(np.hypot(m[0] - x0, m[1] - y0) - R_int)
        return "exterior" if de < di else "interior"

    return _delaunay_dominio(pts, dentro, clasificar)


def malla_heteroestructura(Rc, h, interfaces=(), ajustada=True):
    """Dominio computacional: disco de radio Rc (etiqueta 'borde') que contiene
    las interfaces entre materiales, dadas como círculos [(R, (cx, cy)), ...].

    ajustada=True : se colocan nodos sobre cada interfaz y se retiran los
                    puntos de la grilla a menos de 0.6 h de ella, de modo que
                    los lados de los triángulos siguen la interfaz.
    ajustada=False: malla de fondo uniforme que ignora las interfaces (los
                    triángulos que las cruzan quedan 'cortados').
    """
    for R, (cx, cy) in interfaces:
        if np.hypot(cx, cy) + R >= Rc:
            raise ValueError("Una interfaz no está contenida en el dominio computacional.")
    g = _interior_hex(-Rc, Rc, -Rc, Rc, h)
    ok = np.hypot(g[:, 0], g[:, 1]) < Rc - 0.6 * h
    capas = [_circulo(Rc, h)]
    if ajustada:
        for R, c in interfaces:
            ok &= np.abs(np.hypot(g[:, 0] - c[0], g[:, 1] - c[1]) - R) > 0.6 * h
            capas.append(_circulo(R, h, c))
    pts = np.vstack(capas + [g[ok]])
    dentro = lambda x, y: np.hypot(x, y) < Rc
    malla = _delaunay_dominio(pts, dentro, lambda m: "borde")
    # curvas sobre las que el refinamiento adaptativo proyecta los nodos nuevos
    malla.curvas = [(Rc, (0.0, 0.0))] + (list(interfaces) if ajustada else [])
    return malla


# ======================================================================
# GRÁFICOS
# ======================================================================
def graficar_malla(malla, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    ax.triplot(malla.nodos[:, 0], malla.nodos[:, 1], malla.elems, lw=0.4)
    ax.set_aspect("equal")
    return ax


def graficar_estado(malla, psi, ax=None, densidad=True, titulo=""):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    z = psi ** 2 if densidad else psi
    c = ax.tricontourf(malla.nodos[:, 0], malla.nodos[:, 1], malla.elems, z, 40)
    plt.colorbar(c, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(titulo)
    return ax



# ======================================================================
# PRUEBAS
# ======================================================================
def _test_bessel(R=10.0, h=0.8, k=8):
    """Disco con Dirichlet homogéneo y V=0 vs. ceros de Bessel."""
    from scipy.special import jn_zeros
    coef = HBAR ** 2 / (2 * M_GAAS) * J_M2_A_EV_NM2      # eV nm^2
    ex = []
    for m in range(0, 5):
        for z in jn_zeros(m, 3):
            ex += [coef * (z / R) ** 2] * (1 if m == 0 else 2)  # degeneración
    ex = np.sort(ex)[:k]

    malla = malla_disco(R, h)
    sol = FEMSchrodinger2D(malla, V=0.0, hbar=HBAR, masa=M_GAAS,
                           bc={"borde": ("dirichlet", 0.0)},
                           factor_unidades=J_M2_A_EV_NM2)
    E, _ = sol.resolver_autovalores(k)
    print(f"[Bessel] nodos={malla.n}, elementos={len(malla.elems)}")
    for a, b in zip(E, ex):
        print(f"  E_FEM={a:.5f} eV  E_exacta={b:.5f} eV  err_rel={abs(a-b)/b:.2e}")


def _test_mms(h=0.5):
    """Solución manufacturada en anillo excéntrico: Dirichlet (exterior) +
    Robin (interior), con V(x,y) arbitrario. f se obtiene simbólicamente.
    Se usan hbar=1, masa=1/2 (coef=1) y factor_unidades=1."""
    Re, Ri, x0, y0 = 10.0, 3.0, 2.0, 1.0
    coef = 1.0
    u = sp.cos(sx / 4) * sp.sin(sy / 5) + 1
    V = 0.05 * (sx ** 2 + sy ** 2) + 1
    f = -coef * (u.diff(sx, 2) + u.diff(sy, 2)) + V * u
    # normal exterior a Omega en el agujero: apunta hacia el centro del agujero
    nx = -(sx - x0) / Ri
    ny = -(sy - y0) / Ri
    dun = u.diff(sx) * nx + u.diff(sy) * ny
    alpha = 0.7
    bc = {"exterior": ("dirichlet", u),
          "interior": ("robin", alpha, dun + alpha * u)}
    malla = malla_anillo_asimetrico(Re, Ri, x0, y0, h)
    sol = FEMSchrodinger2D(malla, V=V, hbar=1.0, masa=0.5, bc=bc)
    uh = sol.resolver_estacionario(E=0.0, f=f)
    print(f"[MMS] h={h}, nodos={malla.n}, error L2 = {sol.error_L2(uh, u):.3e}")


if __name__ == "__main__":
    _test_bessel()
    for h in (1.0, 0.5, 0.25):
        _test_mms(h)
