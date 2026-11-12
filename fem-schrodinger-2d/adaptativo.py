"""
adaptativo.py  (Fase 4)
=======================
Refinamiento h-adaptativo para el solver de fem_schrodinger2d.py:

    RESOLVER -> ESTIMAR -> MARCAR -> REFINAR

* estimador()      estimador residual a posteriori por elemento (Ec. 1 del
                   anteproyecto) para cada autopar (E_n, psi_n):
                     eta_T^2 = h_T^2 ||(E_n - V) psi_n||^2_T
                               + 1/2 sum_{e en dT} h_e ||[C grad psi_n . n]||^2_e
                   El salto de C grad psi . n en las aristas interiores es el
                   incumplimiento discreto de la condición de BenDaniel-Duke.
                   Para varios estados se combina eta_T^2 = sum_n eta_{T,n}^2 / E_n
                   (decisión de diseño: el error de E_n es ~ eta_n^2, de modo que
                   dividir por E_n pondera el error relativo de cada nivel).
* marcar_dorfler() conjunto mínimo de elementos que acumula theta del total.
* refinar_nvb()    bisección del vértice más reciente (Funken, Praetorius y
                   Wissgott, 2011): malla conforme, sin nodos colgantes y con
                   un número finito de formas de triángulo. Los nodos nuevos
                   sobre una curva registrada en malla.curvas (borde exterior o
                   interfaces de una malla ajustada) se proyectan sobre ella.
"""
import numpy as np
from fem_schrodinger2d import Malla, geometria, regla_compuesta


# ----------------------------------------------------------------------
# Estructura de aristas
# ----------------------------------------------------------------------
def aristas(malla):
    """Devuelve (E2, ed, vecinos):
    E2 (Na,2) aristas únicas (ordenadas), ed (Ne,3) índice de la arista local
    k del elemento (k=0: nodos 0-1, k=1: 1-2, k=2: 2-0) y vecinos (Na,2) con
    los elementos a cada lado (-1 si es de frontera)."""
    el = malla.elems
    loc = np.stack([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]], axis=1)   # (Ne,3,2)
    pares = np.sort(loc.reshape(-1, 2), axis=1)
    E2, inv = np.unique(pares, axis=0, return_inverse=True)
    inv = inv.ravel()
    ed = inv.reshape(-1, 3)
    elem_de = np.repeat(np.arange(len(el)), 3)
    orden = np.argsort(inv, kind="stable")
    cuenta = np.bincount(inv, minlength=len(E2))
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    vecinos = -np.ones((len(E2), 2), int)
    vecinos[:, 0] = elem_de[orden[inicio]]
    dos = cuenta == 2
    vecinos[dos, 1] = elem_de[orden[inicio[dos] + 1]]
    return E2, ed, vecinos


# ----------------------------------------------------------------------
# Estimador residual
# ----------------------------------------------------------------------
def estimador(sol, E, Psi, nsub=4, por_estado=False, osc=False):
    """eta_T^2 combinado (Ne,) para los autopares (E, Psi) de un
    FEMSchrodinger2D ya resuelto. Si por_estado=True devuelve (Ne, k).
    osc=True añade, en los elementos cortados por una interfaz, el término
    ||(C - C_T) grad psi_h||_T^2 (C_T: valor constante usado en el elemento)."""
    m = sol.malla
    A, b, c = geometria(m)
    X = m.nodos[m.elems]
    k = len(E)
    # --- residuo en el interior: (E - V) psi_h  (div(C grad psi_h) = 0 en P1) ---
    lam, w = regla_compuesta(nsub)
    Q = np.einsum("qi,eid->eqd", lam, X)
    Vq = sol.V(Q[..., 0], Q[..., 1])                              # (Ne,nq)
    psi_q = np.einsum("qi,eik->eqk", lam, Psi[m.elems])            # (Ne,nq,k)
    r2 = A[:, None] * np.einsum("q,eqk->ek", w, ((E[None, None, :] - Vq[..., None]) * psi_q) ** 2)
    lados = np.linalg.norm(X[:, [1, 2, 0]] - X, axis=2)           # (Ne,3)
    hT = lados.max(1)
    eta2 = hT[:, None] ** 2 * r2
    # --- saltos del flujo C grad psi . n en aristas interiores ---
    E2, ed, vec = aristas(m)
    grad = np.stack([np.einsum("ei,eik->ek", b, Psi[m.elems]),
                     np.einsum("ei,eik->ek", c, Psi[m.elems])], axis=1) / (2 * A[:, None, None])
    flujo = sol.Ce[:, None, None] * grad                          # (Ne,2,k)
    inte = vec[:, 1] >= 0
    t = m.nodos[E2[inte, 1]] - m.nodos[E2[inte, 0]]
    le = np.linalg.norm(t, axis=1)
    n = np.stack([t[:, 1], -t[:, 0]], 1) / le[:, None]
    T1, T2 = vec[inte, 0], vec[inte, 1]
    salto = np.einsum("ad,adk->ak", n, flujo[T1] - flujo[T2])      # (Na_int,k)
    contrib = 0.5 * le[:, None] * (salto ** 2 * le[:, None])       # h_e * ||J||^2_e, mitad a cada lado
    np.add.at(eta2, T1, contrib)
    np.add.at(eta2, T2, contrib)
    if osc and sol.cortados.any():
        kc = sol.cortados
        lam2, w2 = regla_compuesta(sol.nsub_corte)
        Q2 = np.einsum("qi,eid->eqd", lam2, X[kc])
        d2 = A[kc] * ((sol.C(Q2[..., 0], Q2[..., 1]) - sol.Ce[kc, None]) ** 2 @ w2)
        eta2[kc] += d2[:, None] * (grad[kc] ** 2).sum(1)
    if por_estado:
        return eta2
    return eta2 @ (1.0 / np.maximum(E, 1e-30))


def marcar_dorfler(eta2, theta=0.5):
    """Índices del menor conjunto de elementos con sum eta2 >= theta * total."""
    orden = np.argsort(eta2)[::-1]
    acum = np.cumsum(eta2[orden])
    n = int(np.searchsorted(acum, theta * acum[-1]) + 1)
    return orden[:n]


# ----------------------------------------------------------------------
# Bisección del vértice más reciente (NVB)
# ----------------------------------------------------------------------
def preparar_nvb(malla):
    """Rota cada triángulo (sin cambiar su orientación) para que su arista de
    refinamiento (nodos 0-1) sea la más larga. Se llama una vez sobre la malla
    inicial; después NVB mantiene la convención."""
    X = malla.nodos[malla.elems]
    lados = np.linalg.norm(X[:, [1, 2, 0]] - X, axis=2)            # lado k: nodos k, k+1
    k = lados.argmax(1)
    idx = (np.arange(3)[None, :] + k[:, None]) % 3
    malla.elems = np.take_along_axis(malla.elems, idx, axis=1)
    malla.nvb = True
    return malla


def _proyectar(malla, P, a, b):
    """Proyecta los puntos medios P de las aristas (a,b) cuyos dos extremos
    están sobre una curva registrada (círculo R, centro c)."""
    for R, c in getattr(malla, "curvas", []):
        c = np.asarray(c, float)
        tol = 1e-9 * max(1.0, R)
        da = np.abs(np.linalg.norm(malla.nodos[a] - c, axis=1) - R)
        db = np.abs(np.linalg.norm(malla.nodos[b] - c, axis=1) - R)
        sobre = (da < tol) & (db < tol)
        if sobre.any():
            d = P[sobre] - c
            P[sobre] = c + R * d / np.linalg.norm(d, axis=1)[:, None]
    return P


def refinar_nvb(malla, marcados):
    """Refina los elementos marcados (y los necesarios para conformidad) por
    bisección del vértice más reciente. Devuelve una Malla nueva."""
    if not getattr(malla, "nvb", False):
        preparar_nvb(malla)
    el = malla.elems
    E2, ed, vec = aristas(malla)
    mk = np.zeros(len(E2), bool)
    mk[ed[marcados, 0]] = True
    # cierre: si un elemento tiene alguna arista marcada, su arista de
    # refinamiento también debe estarlo
    while True:
        falta = mk[ed].any(1) & ~mk[ed[:, 0]]
        if not falta.any():
            break
        mk[ed[falta, 0]] = True
    # nodos nuevos en los puntos medios de las aristas marcadas
    idm = np.where(mk)[0]
    a, b = E2[idm, 0], E2[idm, 1]
    P = _proyectar(malla, 0.5 * (malla.nodos[a] + malla.nodos[b]), a, b)
    medio = -np.ones(len(E2), int)
    medio[idm] = malla.n + np.arange(len(idm))
    nodos = np.vstack([malla.nodos, P])

    z1, z2, z3 = el[:, 0], el[:, 1], el[:, 2]
    r = mk[ed[:, 0]]
    sigue = el[~r]
    z1, z2, z3 = z1[r], z2[r], z3[r]
    m0 = medio[ed[r, 0]]
    q = medio[ed[r, 1]]          # punto medio de z2-z3 (o -1)
    p = medio[ed[r, 2]]          # punto medio de z3-z1 (o -1)
    # bisección por la arista de refinamiento: (z3,z1,m0) y (z2,z3,m0)
    h1 = np.stack([z3, z1, m0], 1)
    h2 = np.stack([z2, z3, m0], 1)
    # segunda bisección de cada hijo por su propia arista de refinamiento
    bp, aq = p >= 0, q >= 0
    hijos = [sigue,
             h1[~bp], np.stack([m0, z3, p], 1)[bp], np.stack([z1, m0, p], 1)[bp],
             h2[~aq], np.stack([m0, z2, q], 1)[aq], np.stack([z3, m0, q], 1)[aq]]
    nuevos = np.vstack(hijos)

    # aristas de frontera: las bisecadas se reemplazan por sus dos mitades
    fr = np.sort(malla.aristas, axis=1)
    ids = np.array([_buscar(E2, f) for f in fr], int) if len(fr) else np.zeros(0, int)
    ar_n, et_n = [], []
    for (u, v), i, et in zip(fr, ids, malla.etiquetas):
        if mk[i]:
            ar_n += [(u, medio[i]), (medio[i], v)]
            et_n += [et, et]
        else:
            ar_n.append((u, v))
            et_n.append(et)
    nueva = Malla(nodos, nuevos, ar_n, et_n)
    nueva.curvas = getattr(malla, "curvas", [])
    nueva.nvb = True
    return nueva


def _buscar(E2, par):
    """Índice de la arista 'par' (ordenada) en E2 (ordenado lexicográficamente)."""
    i = np.searchsorted(E2[:, 0], par[0], side="left")
    j = np.searchsorted(E2[:, 0], par[0], side="right")
    return i + np.searchsorted(E2[i:j, 1], par[1])


def calidad(malla):
    """Ángulo mínimo (grados) de la malla: debe mantenerse acotado con NVB."""
    X = malla.nodos[malla.elems]
    ang = []
    for k in range(3):
        u = X[:, (k + 1) % 3] - X[:, k]
        v = X[:, (k + 2) % 3] - X[:, k]
        cos = np.einsum("ed,ed->e", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return float(np.min(ang))
