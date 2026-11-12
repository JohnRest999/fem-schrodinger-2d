"""
fem_schrodinger2d.py  (versión 2, Fase 3)
=========================================
Solucionador FEM (Galerkin, funciones sombrero P1) para la ecuación de
Schrödinger 2D con masa efectiva y potencial dependientes de la posición:

        -div( C(x,y) grad psi ) + V(x,y) psi = E psi      en Omega,
        C(x,y) = hbar^2 / (2 m*(x,y))

Las condiciones de BenDaniel-Duke en una interfaz entre materiales
(psi y C dpsi/dn continuas) son NATURALES en la forma débil: no se imponen.

Condiciones de frontera POR ETIQUETA (normal n exterior a Omega):
    Dirichlet :  psi = g
    Neumann   :  C dpsi/dn = C h
    Robin     :  C (dpsi/dn + alpha psi) = C g

Cambios respecto a la versión 1
-------------------------------
* Ensamblaje VECTORIZADO (sin bucle de Python por elemento). Las fórmulas
  cerradas que usa se comparan con las matrices deducidas simbólicamente con
  sympy en _test_formulas() (deben coincidir a precisión de máquina).
* C puede variar por elemento (masa efectiva variable).
* Dos maneras de integrar V y C (argumento `integracion`):
    "nodal"      V interpolado con las funciones P1 (exacto con el tensor
                 T_ijk = int N_i N_j N_k); C evaluado en el centroide.
                 Es el método de la versión 1.
    "cuadratura" regla compuesta: cada triángulo se divide en nsub^2
                 subtriángulos y V y C se evalúan en sus centroides. En los
                 elementos CORTADOS por una interfaz (donde V o C saltan) se
                 usa una subdivisión más fina (nsub_corte) y C se promedia con
                 la media ARMÓNICA (media_C="armonica", por defecto): como el
                 flujo C dpsi/dn es continuo a través de la interfaz, los dos
                 materiales actúan como resistencias en serie. En el disco con
                 malla no ajustada esto recupera casi el orden óptimo
                 (ver barrera_finita.py, experimento E2).
* Compatible con pruebas.py y resultados.py: acepta hbar, masa y
  factor_unidades, y exporta los símbolos x, y de sympy.

Unidades: el solver no impone unidades. Con hbar, masa en SI y
factor_unidades = 1/(q_e*1e-18) trabaja en eV y nm (como en el informe de
avance). Con coef = mu(x,y) = m_ref/m*(x,y), V/E0 y longitudes en unidades de
L trabaja en unidades naturales (ver energia_natural()).
"""
import numpy as np
import sympy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh, spsolve

# Símbolos públicos para escribir V, f, g como expresiones sympy
x, y = sp.symbols("x y", real=True)
_x, _y = x, y                     # alias de la versión 1

HBAR2_2ME_EV_NM2 = 0.0380998212   # hbar^2/(2 m_e) en eV nm^2


def energia_natural(L_nm, m_rel=0.067):
    """Escala de energía E0 = hbar^2/(2 m L^2) en eV, para L en nm y
    m = m_rel * m_e. En unidades naturales (x/L, E/E0), un material con masa
    m_rel tiene operador -Laplaciano."""
    return HBAR2_2ME_EV_NM2 / (m_rel * L_nm ** 2)


# ======================================================================
# 1. DEDUCCIÓN SIMBÓLICA (sympy) DE LAS MATRICES ELEMENTALES
# ======================================================================
def derivar_matrices_elementales():
    """
    Triángulo (x1,y1),(x2,y2),(x3,y3) antihorario, mapeado desde el triángulo
    de referencia con N1 = 1-xi-eta, N2 = xi, N3 = eta. Devuelve:
      Ke[i,j]  = int grad N_i . grad N_j       (simbólica, C = 1)
      Me[i,j]  = int N_i N_j                   (simbólica)
      T[i,j,k] = int_ref N_i N_j N_k           (numérica, para V interpolado)
      Ml[i,j]  = int_ref N_i N_j               (numérica)
      Mb       = int_arista N_a N_b ds         (2x2, simbólica en L)
    """
    x1, y1, x2, y2, x3, y3 = sp.symbols("x1 y1 x2 y2 x3 y3", real=True)
    xi, eta = sp.symbols("xi eta", real=True)
    N = [1 - xi - eta, xi, eta]
    X = sum(Ni * v for Ni, v in zip(N, (x1, x2, x3)))
    Y = sum(Ni * v for Ni, v in zip(N, (y1, y2, y3)))
    J = sp.Matrix([[X.diff(xi), X.diff(eta)], [Y.diff(xi), Y.diff(eta)]])
    detJ = sp.simplify(J.det())                     # = 2*Area
    Jit = sp.simplify(J.inv().T)
    grads = [sp.simplify(Jit * sp.Matrix([Ni.diff(xi), Ni.diff(eta)])) for Ni in N]

    def integ_ref(expr):
        return sp.integrate(sp.integrate(expr, (eta, 0, 1 - xi)), (xi, 0, 1))

    Ke = sp.Matrix(3, 3, lambda i, j: sp.simplify(integ_ref(grads[i].dot(grads[j])) * detJ))
    Me = sp.Matrix(3, 3, lambda i, j: sp.simplify(integ_ref(N[i] * N[j]) * detJ))
    Ml = np.array([[float(integ_ref(N[i] * N[j])) for j in range(3)] for i in range(3)])
    T = np.array([[[float(integ_ref(N[i] * N[j] * N[k])) for k in range(3)]
                   for j in range(3)] for i in range(3)])
    s, L = sp.symbols("s L", positive=True)
    nn = [1 - s / L, s / L]
    Mb = sp.Matrix(2, 2, lambda a, b: sp.integrate(nn[a] * nn[b], (s, 0, L)))
    return Ke, Me, T, Ml, Mb, L, detJ / 2, (x1, y1, x2, y2, x3, y3)


_Ke_s, _Me_s, _T, _Ml, _Mb_s, _L_s, _A_s, _sim = derivar_matrices_elementales()
_Ke_f = sp.lambdify(_sim, _Ke_s, "numpy")
_Me_f = sp.lambdify(_sim, _Me_s, "numpy")
_Mb_ref = np.array(_Mb_s.subs(_L_s, 1), float)      # Mb = L * _Mb_ref


# ======================================================================
# 2. MALLA
# ======================================================================
class Malla:
    """nodos (N,2), elems (Ne,3) antihorarios, aristas (Nb,2), etiquetas (Nb,)"""

    def __init__(self, nodos, elems, aristas, etiquetas, clasificar=None):
        self.nodos = np.asarray(nodos, float)
        self.elems = np.asarray(elems, int)
        self.aristas = np.asarray(aristas, int).reshape(-1, 2)
        self.etiquetas = np.asarray(etiquetas, object)
        self.clasificar = clasificar      # se reutiliza al refinar la malla

    @property
    def n(self):
        return len(self.nodos)

    def nodos_de(self, etiqueta):
        return np.unique(self.aristas[self.etiquetas == etiqueta].ravel())

    @classmethod
    def desde_triangulos(cls, nodos, elems, clasificar):
        """Orienta los triángulos, detecta las aristas de frontera y las
        etiqueta con clasificar(punto_medio) -> str."""
        nodos = np.asarray(nodos, float)
        elems = np.asarray(elems, int).copy()
        p = nodos[elems]
        a2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
              - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        elems[a2 < 0] = elems[a2 < 0][:, [0, 2, 1]]
        ar = np.sort(np.vstack([elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]]), axis=1)
        u, c = np.unique(ar, axis=0, return_counts=True)
        ar = u[c == 1]
        et = [clasificar(0.5 * (nodos[a] + nodos[b])) for a, b in ar]
        return cls(nodos, elems, ar, et, clasificar)


# ======================================================================
# 3. UTILIDADES
# ======================================================================
def como_funcion(f):
    """Número / callable(x,y) / expresión sympy  ->  f(x,y) vectorizada."""
    if f is None:
        return lambda x_, y_: np.zeros(np.shape(x_))
    if isinstance(f, sp.Basic):
        fl = sp.lambdify((x, y), f, "numpy")
        return lambda x_, y_: np.broadcast_to(np.asarray(fl(x_, y_), float), np.shape(x_)).copy()
    if callable(f):
        return lambda x_, y_: np.broadcast_to(np.asarray(f(x_, y_), float), np.shape(x_)).copy()
    c = float(f)
    return lambda x_, y_: np.full(np.shape(x_), c)


def regla_compuesta(nsub):
    """Coordenadas baricéntricas (nq,3) de los centroides de los nsub^2
    subtriángulos de igual área, y pesos relativos (todos 1/nsub^2)."""
    pts = [((i + 1 / 3) / nsub, (j + 1 / 3) / nsub) for i in range(nsub) for j in range(nsub - i)]
    pts += [((i + 2 / 3) / nsub, (j + 2 / 3) / nsub) for i in range(nsub) for j in range(nsub - i - 1)]
    lam = np.array([(1 - a - b, a, b) for a, b in pts])
    return lam, np.full(len(lam), 1.0 / len(lam))


def geometria(malla):
    """Áreas A (Ne,) y coeficientes b, c (Ne,3): grad N_i = (b_i, c_i)/(2A)."""
    X = malla.nodos[malla.elems]
    b = np.stack([X[:, 1, 1] - X[:, 2, 1], X[:, 2, 1] - X[:, 0, 1], X[:, 0, 1] - X[:, 1, 1]], 1)
    c = np.stack([X[:, 2, 0] - X[:, 1, 0], X[:, 0, 0] - X[:, 2, 0], X[:, 1, 0] - X[:, 0, 0]], 1)
    A = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    return A, b, c


# ======================================================================
# 4. SOLVER
# ======================================================================
class FEMSchrodinger2D:
    """
    Parameters
    ----------
    malla : Malla
    V     : número | callable(x,y) | expresión sympy
    coef  : C(x,y) = hbar^2/(2 m*): número | callable | sympy. Si se da, manda.
    hbar, masa, factor_unidades : alternativa a coef:
            C = factor_unidades * hbar^2 / (2 * masa(x,y)); masa puede ser
            número, callable o sympy (masa efectiva variable).
            Sin coef ni hbar/masa: GaAs en eV nm^2.
    bc    : {etiqueta: ('dirichlet', g) | ('neumann', h) | ('robin', alpha, g)}
            Etiquetas sin entrada -> Neumann homogénea.
    integracion : "nodal" (versión 1) | "cuadratura"
    nsub, nsub_corte : subdivisiones de la regla compuesta en elementos
            normales y en elementos cortados por una interfaz.
    tol_salto : un elemento se considera cortado si V o C varían en él más que
            tol_salto * (rango global de V o de C). Con potenciales suaves el
            criterio no se activa.
    media_C : "armonica" | "aritmetica": promedio de C en los elementos cortados.
    """

    def __init__(self, malla, V=0.0, hbar=None, masa=None, bc=None,
                 factor_unidades=1.0, coef=None, integracion="nodal",
                 nsub=4, nsub_corte=12, tol_salto=0.25, media_C="armonica"):
        self.malla = malla
        self.V = como_funcion(V)
        if coef is not None:
            self.C = como_funcion(coef)
        elif hbar is not None and masa is not None:
            mf = como_funcion(masa)
            k = factor_unidades * hbar ** 2 / 2.0
            self.C = lambda x_, y_: k / mf(x_, y_)
        else:
            self.C = como_funcion(HBAR2_2ME_EV_NM2 / 0.067)
        self.bc = bc or {}
        if integracion not in ("nodal", "cuadratura"):
            raise ValueError("integracion debe ser 'nodal' o 'cuadratura'")
        self.integracion = integracion
        self.nsub, self.nsub_corte, self.tol_salto = nsub, nsub_corte, tol_salto
        if media_C not in ("armonica", "aritmetica"):
            raise ValueError("media_C debe ser 'armonica' o 'aritmetica'")
        self.media_C = media_C
        self.ensamblar()

    # ------------------------------------------------------------------
    def _integrar_V_C(self, A, X):
        """Ve (Ne,3,3) = int_e V N_i N_j  y  Ce (Ne,) = promedio de C en e."""
        if self.integracion == "nodal":
            Vn = self.V(self.malla.nodos[:, 0], self.malla.nodos[:, 1])
            Ve = 2 * A[:, None, None] * np.einsum("ijk,ek->eij", _T, Vn[self.malla.elems])
            cen = X.mean(1)
            self.cortados = np.zeros(len(A), bool)
            return Ve, self.C(cen[:, 0], cen[:, 1])

        lam, w = regla_compuesta(self.nsub)
        Q = np.einsum("qi,eid->eqd", lam, X)
        Vq, Cq = self.V(Q[..., 0], Q[..., 1]), self.C(Q[..., 0], Q[..., 1])
        # detección de elementos cortados (puntos de cuadratura + vértices)
        Vall = np.hstack([Vq, self.V(X[..., 0], X[..., 1])])
        Call = np.hstack([Cq, self.C(X[..., 0], X[..., 1])])
        rV, rC = (np.ptp(Vall) or 1.0), (np.ptp(Call) or 1.0)
        cort = (np.ptp(Vall, 1) > self.tol_salto * rV) | (np.ptp(Call, 1) > self.tol_salto * rC)
        Ve = A[:, None, None] * np.einsum("eq,q,qi,qj->eij", Vq, w, lam, lam)
        Ce = Cq @ w
        if cort.any():
            lam2, w2 = regla_compuesta(self.nsub_corte)
            Q2 = np.einsum("qi,eid->eqd", lam2, X[cort])
            Ve[cort] = A[cort, None, None] * np.einsum(
                "eq,q,qi,qj->eij", self.V(Q2[..., 0], Q2[..., 1]), w2, lam2, lam2)
            C2 = self.C(Q2[..., 0], Q2[..., 1])
            Ce[cort] = 1.0 / ((1.0 / C2) @ w2) if self.media_C == "armonica" else C2 @ w2
        self.cortados = cort
        return Ve, Ce

    # ------------------------------------------------------------------
    def ensamblar(self):
        m = self.malla
        n, E = m.n, m.elems
        A, b, c = geometria(m)
        if np.any(A <= 0):
            raise ValueError("Hay triángulos con área no positiva.")
        Ve, Ce = self._integrar_V_C(A, m.nodos[E])
        self.Ce = Ce
        Kg = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * A[:, None, None])
        Me = A[:, None, None] / 12.0 * (np.ones((3, 3)) + np.eye(3))
        I = np.repeat(E, 3, axis=1).ravel()
        J = np.tile(E, (1, 3)).ravel()
        mk = lambda v: sps.coo_matrix((v.ravel(), (I, J)), shape=(n, n)).tocsr()
        self.K, self.M, self.Vm = mk(Ce[:, None, None] * Kg), mk(Me), mk(Ve)

        # ----- bordes (Neumann / Robin), vectorizado por etiqueta -----
        self.b_borde = np.zeros(n)
        Ri, Rj, Rv = [], [], []
        for et in (np.unique(m.etiquetas) if len(m.etiquetas) else []):
            spec = self.bc.get(et, ("neumann", 0.0))
            if spec[0] == "dirichlet":
                continue
            ar = m.aristas[m.etiquetas == et]
            pa, pb = m.nodos[ar[:, 0]], m.nodos[ar[:, 1]]
            Lr = np.linalg.norm(pb - pa, axis=1)
            mid = 0.5 * (pa + pb)
            Cm = self.C(mid[:, 0], mid[:, 1])
            Mb = Lr[:, None, None] * _Mb_ref                 # (Nb,2,2)
            xs, ys = m.nodos[ar, 0], m.nodos[ar, 1]          # (Nb,2)
            if spec[0] == "neumann":
                h = como_funcion(spec[1])(xs, ys)
                np.add.at(self.b_borde, ar, Cm[:, None] * np.einsum("eab,eb->ea", Mb, h))
            elif spec[0] == "robin":
                al = como_funcion(spec[1])(xs, ys).mean(1)
                g = como_funcion(spec[2])(xs, ys)
                np.add.at(self.b_borde, ar, Cm[:, None] * np.einsum("eab,eb->ea", Mb, g))
                Ri.append(np.repeat(ar, 2, 1).ravel())
                Rj.append(np.tile(ar, (1, 2)).ravel())
                Rv.append(((Cm * al)[:, None, None] * Mb).ravel())
            else:
                raise ValueError(f"Tipo de condición desconocido: {spec[0]}")
        if Ri:
            self.R = sps.coo_matrix((np.concatenate(Rv), (np.concatenate(Ri), np.concatenate(Rj))),
                                    shape=(n, n)).tocsr()
        else:
            self.R = sps.csr_matrix((n, n))

    # ------------------------------------------------------------------
    def _dirichlet(self):
        m = self.malla
        fijo = np.zeros(m.n, bool)
        val = np.zeros(m.n)
        for et, spec in self.bc.items():
            if spec[0] == "dirichlet":
                nd = m.nodos_de(et)
                fijo[nd] = True
                val[nd] = como_funcion(spec[1])(m.nodos[nd, 0], m.nodos[nd, 1])
        return fijo, val

    def resolver_autovalores(self, k=6, sigma=None):
        """(K + V + R) psi = E M psi, Dirichlet homogéneo eliminado de la
        incógnita. Devuelve (E, Psi) con int |psi_n|^2 = 1."""
        fijo, _ = self._dirichlet()
        libre = ~fijo
        A = (self.K + self.Vm + self.R).tocsc()[libre][:, libre]
        Mf = self.M.tocsc()[libre][:, libre]
        if sigma is None:
            Vmin = float(np.min(self.V(self.malla.nodos[:, 0], self.malla.nodos[:, 1])))
            sigma = min(0.0, Vmin) - 1e-3 * (1 + abs(Vmin))
        E, Xv = eigsh(A, k=k, M=Mf, sigma=sigma, which="LM")
        o = np.argsort(E)
        E, Xv = E[o], Xv[:, o]
        Psi = np.zeros((self.malla.n, k))
        Psi[libre] = Xv
        Psi /= np.sqrt(np.einsum("ik,ik->k", Psi, self.M @ Psi))
        return E, Psi

    def resolver_estacionario(self, E=0.0, f=None):
        """(-div(C grad) + V - E) psi = f con las BC dadas (f interpolada en P1)."""
        m = self.malla
        fijo, val = self._dirichlet()
        libre = ~fijo
        fn = como_funcion(f)(m.nodos[:, 0], m.nodos[:, 1])
        b = self.M @ fn + self.b_borde
        A = (self.K + self.Vm + self.R - E * self.M).tocsr()
        u = val.copy()
        u[libre] = spsolve(A[libre][:, libre].tocsc(), b[libre] - A[libre][:, fijo] @ val[fijo])
        return u

    def error_L2(self, u, u_exacta):
        ue = como_funcion(u_exacta)(self.malla.nodos[:, 0], self.malla.nodos[:, 1])
        e = u - ue
        return float(np.sqrt(e @ (self.M @ e)))


# ======================================================================
# 5. PRUEBA: fórmulas vectorizadas == matrices deducidas con sympy
# ======================================================================
def _test_formulas(n=200, semilla=0):
    rng = np.random.default_rng(semilla)
    peor_K = peor_M = 0.0
    for _ in range(n):
        P = rng.normal(size=(3, 2))
        if (P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]) < 0:
            P = P[[0, 2, 1]]
        m = Malla(P, [[0, 1, 2]], np.zeros((0, 2)), [])
        A, b, c = geometria(m)
        Kv = (np.outer(b[0], b[0]) + np.outer(c[0], c[0])) / (4 * A[0])
        Mv = A[0] / 12 * (np.ones((3, 3)) + np.eye(3))
        Ks = np.array(_Ke_f(*P.ravel()), float)
        Ms = np.array(_Me_f(*P.ravel()), float)
        peor_K = max(peor_K, np.abs(Kv - Ks).max() / np.abs(Ks).max())
        peor_M = max(peor_M, np.abs(Mv - Ms).max() / np.abs(Ms).max())
    print(f"[fórmulas] máx. error relativo  K: {peor_K:.1e}   M: {peor_M:.1e}")
    return peor_K, peor_M


if __name__ == "__main__":
    _test_formulas()
