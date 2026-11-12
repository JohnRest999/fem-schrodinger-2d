"""
fase5_anillo.py  (Fase 5)
=========================
Estudio físico del anillo cuántico excéntrico de GaAs/Al0.3Ga0.7As con los
parámetros de Vinasco et al., Sci. Rep. 9, 1427 (2019):
R1 = 12 nm (agujero), R2 = 28 nm, V0 = 262 meV, m_in = 0.067, m_out = 0.093,
excentricidad w = x0 (desplazamiento del agujero en x), B = 0.

Unidades naturales: L = R2 = 28 nm, E0 = hbar^2/(2 m_in L^2).

P1  Espectro E_n(w), w = 0..10 nm, 10 niveles, malla adaptativa no ajustada
    (barrera finita) y malla uniforme ajustada (pozo infinito).
P2  Ruptura de la degeneración: desdoblamiento de los dobletes |m| = 1 y 2 en
    función de w sobre una malla uniforme hexagonal fija (casi simétrica bajo
    rotaciones de 60°, lo que protege los dobletes m = ±1, ±2 a w = 0).
    Teoría de perturbaciones: Delta_|m| ~ w^(2|m|).
P3  Localización del estado base: centroide <x>, probabilidad en la barrera
    y en la garganta, en función de w.
P4  Figuras.

Clasificación de estados: paridad bajo la reflexión y -> -y (el problema es
simétrico respecto del eje x). Par (+1): tipo s, p_x, d_{x^2-y^2}; impar (-1):
tipo p_y, d_xy.

Uso:  python fase5_anillo.py P1      (o P2, P3, P4; sin argumentos: todo)
Salida: figuras_fase5/*.png y figuras_fase5/resultados_fase5.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

from fem_schrodinger2d import FEMSchrodinger2D, energia_natural
from pruebas import malla_heteroestructura, malla_anillo_asimetrico
from adaptativo import estimador, marcar_dorfler, refinar_nvb
from barrera_finita import V0_EV, M_IN, MU_OUT

os.makedirs("figuras_fase5", exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
ARCH = "figuras_fase5/resultados_fase5.json"
res = json.load(open(ARCH)) if os.path.exists(ARCH) else {}

# ----------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------
L = 28.0                                  # nm (= R2)
E0 = energia_natural(L, M_IN)             # eV
MEV = 1e3 * E0                            # factor: unidades naturales -> meV
v0 = V0_EV / E0
R1 = 12.0 / L
RC = 2.0                                  # dominio computacional r = 2 R2 (como Vinasco)
K = 10                                    # niveles
WS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]   # nm
N_ADAPT = 8.0e4                           # nodos de la malla adaptativa final


def dentro_anillo(w_nm):
    c = w_nm / L
    return lambda x, y: (np.hypot(x, y) < 1.0) & (np.hypot(x - c, y) > R1)


def problema(w_nm):
    d = dentro_anillo(w_nm)
    V = lambda x, y: np.where(d(x, y), 0.0, v0)
    C = lambda x, y: np.where(d(x, y), 1.0, MU_OUT)
    return V, C


def resolver(m, V, C, k=K, bc=None):
    s = FEMSchrodinger2D(m, V=V, coef=C, bc=bc or {"borde": ("dirichlet", 0.0)},
                         integracion="cuadratura")
    E, Psi = s.resolver_autovalores(k, sigma=-1e-2)
    return s, E, Psi


def adaptativo(V, C, n_max=N_ADAPT, theta=0.5, h0=2.0 / L):
    m = malla_heteroestructura(RC, h0, [], ajustada=False)
    while True:
        s, E, Psi = resolver(m, V, C)
        eta2 = estimador(s, E, Psi)
        if m.n >= n_max:
            return m, s, E, Psi, float(np.sqrt(eta2.sum()))
        m = refinar_nvb(m, marcar_dorfler(eta2, theta))


# ----------------------------------------------------------------------
# Paridad bajo y -> -y
# ----------------------------------------------------------------------
def paridades(s, E, Psi, tol_deg=1e-4):
    """Devuelve (Psi', paridad) donde, dentro de cada grupo (casi) degenerado,
    los estados se rotan para que sean autoestados de la reflexión."""
    m = s.malla
    tri = Triangulation(m.nodos[:, 0], m.nodos[:, 1], m.elems)

    def refl(v):
        f = LinearTriInterpolator(tri, v)
        r = np.asarray(f(m.nodos[:, 0], -m.nodos[:, 1]).filled(0.0))
        return r

    Psi = Psi.copy()
    par = np.zeros(len(E))
    i = 0
    while i < len(E):
        j = i + 1
        while j < len(E) and abs(E[j] - E[i]) / abs(E[i]) < tol_deg:
            j += 1
        G = Psi[:, i:j]
        RG = np.column_stack([refl(G[:, a]) for a in range(j - i)])
        P = G.T @ (s.M @ RG)
        P = 0.5 * (P + P.T)
        val, vec = np.linalg.eigh(P)
        Psi[:, i:j] = G @ vec
        par[i:j] = val
        i = j
    return Psi, par


def estado_par_impar(E, par):
    """Índices de los estados pares e impares, en orden de energía."""
    return np.where(par > 0)[0], np.where(par < 0)[0]


# ======================================================================
# P1: espectro E_n(w)
# ======================================================================
def parte_P1(t_max=240.0):
    """Incremental: calcula los w que aún no están en el JSON, guarda tras cada
    uno y se detiene antes de t_max segundos (límite de tiempo por ejecución)."""
    print("P1: espectro en función de w")
    t_ini = time.time()
    P1 = res.setdefault("P1", dict(L_nm=L, E0_meV=MEV, filas=[]))
    hechos = {f["w"] for f in P1["filas"]}
    for w in WS:
        if w in hechos:
            continue
        if time.time() - t_ini > t_max:
            print("  (tiempo agotado; vuelva a ejecutar P1 para continuar)")
            return
        t0 = time.time()
        V, C = problema(w)
        m, s, E, Psi, eta = adaptativo(V, C)
        Psi, par = paridades(s, E, Psi)
        # pozo infinito: malla uniforme ajustada al anillo, Dirichlet en ambos bordes
        mi = malla_anillo_asimetrico(1.0, R1, w / L, 0.0, 0.25 / L)
        si = FEMSchrodinger2D(mi, V=0.0, coef=1.0,
                              bc={"exterior": ("dirichlet", 0.0), "interior": ("dirichlet", 0.0)})
        Ei, _ = si.resolver_autovalores(K)
        P1["filas"].append(dict(w=w, nodos=m.n, eta=eta, E_meV=(MEV * E).tolist(), paridad=par.tolist(),
                                E_inf_meV=(MEV * Ei).tolist(), nodos_inf=mi.n))
        P1["filas"].sort(key=lambda f: f["w"])
        json.dump(res, open(ARCH, "w"), indent=1)
        print(f"  w={w:4.1f} nm  N={m.n:6d}  E(meV)={np.round(MEV*E[:6],3)}  "
              f"paridad={np.sign(par[:6]).astype(int)}  ({time.time()-t0:.0f}s)", flush=True)


def parte_P1c():
    """Control de precisión de P1: w = 5 nm con una malla adaptativa más fina."""
    V, C = problema(5.0)
    m2, s2, E2, _, _ = adaptativo(V, C, n_max=1.6e5)
    E1 = np.array([f for f in res["P1"]["filas"] if f["w"] == 5.0][0]["E_meV"])
    d = np.abs(MEV * E2 - E1) / (MEV * E2)
    res["P1"]["control_w5"] = dict(N_fino=m2.n, dif_rel=d.tolist(), dif_rel_max=float(d.max()))
    print(f"  control w=5: N={m2.n}  máx |E_80k - E_fino|/E = {d.max():.1e}")


# ======================================================================
# P2: desdoblamiento de los dobletes
# ======================================================================
def parte_P2():
    print("P2: desdoblamiento de dobletes en malla hexagonal fija")
    WS2 = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    m = malla_heteroestructura(RC, 0.35 / L, [], ajustada=False)    # misma malla para todo w
    filas = []
    for w in WS2:
        V, C = problema(w)
        s, E, Psi = resolver(m, V, C, k=8)
        Psi, par = paridades(s, E, Psi)
        pa, im = estado_par_impar(E, par)
        # doblete |m|=1: 2.º par (p_x) y 1.º impar (p_y); |m|=2: 3.º par y 2.º impar
        d1 = MEV * (E[im[0]] - E[pa[1]])
        d2 = MEV * (E[im[1]] - E[pa[2]])
        filas.append(dict(w=w, E_meV=(MEV * E).tolist(), paridad=par.tolist(), d1_meV=float(d1), d2_meV=float(d2)))
        print(f"  w={w:4.2f}  Δ(|m|=1)={d1:+.3e} meV  Δ(|m|=2)={d2:+.3e} meV", flush=True)
    piso1, piso2 = abs(filas[0]["d1_meV"]), abs(filas[0]["d2_meV"])
    # pendientes log-log con w <= 2 nm (régimen perturbativo) y por encima del piso
    def pendiente(clave, piso):
        w = np.array([f["w"] for f in filas if 0 < f["w"] <= 2.0])
        d = np.array([abs(f[clave]) for f in filas if 0 < f["w"] <= 2.0])
        ok = d > 10 * piso
        return float(np.polyfit(np.log(w[ok]), np.log(d[ok]), 1)[0]) if ok.sum() >= 2 else None, int(ok.sum())
    p1, n1 = pendiente("d1_meV", piso1)
    p2, n2 = pendiente("d2_meV", piso2)
    # pendientes locales entre valores consecutivos de w (solo por encima de 10x el piso)
    loc = {}
    for clave, piso in (("d1_meV", piso1), ("d2_meV", piso2)):
        w = np.array([f["w"] for f in filas]); d = np.abs([f[clave] for f in filas])
        loc[clave] = [(float(w[i]), float(w[i + 1]), float(np.log(d[i + 1] / d[i]) / np.log(w[i + 1] / w[i])))
                      for i in range(1, len(w) - 1) if d[i] > 10 * piso]
        print(f"  pendientes locales {clave}: " +
              ", ".join(f"[{a:g},{b:g}]: {p:.2f}" for a, b, p in loc[clave]))
    res["P2"] = dict(h_nm=0.35, nodos=m.n, filas=filas, piso_meV=[piso1, piso2],
                     pendiente_m1=p1, puntos_m1=n1, pendiente_m2=p2, puntos_m2=n2,
                     pendientes_locales=loc)
    print(f"  piso a w=0: {piso1:.1e}, {piso2:.1e} meV;  pendientes: |m|=1 -> {p1} ({n1} pts), "
          f"|m|=2 -> {p2} ({n2} pts)   [teoría: 2 y 4]")


# ======================================================================
# P3: localización del estado base
# ======================================================================
def parte_P3():
    print("P3: localización")
    filas = []
    for w in WS:
        V, C = problema(w)
        m, s, E, Psi, _ = adaptativo(V, C, n_max=4.0e4)
        Psi, _ = paridades(s, E, Psi)        # base de paridad en grupos degenerados (w = 0)
        X = m.nodos
        A = s.M
        rho = Psi[:, :4] ** 2
        uno = np.ones(m.n)
        d = dentro_anillo(w)(X[:, 0], X[:, 1]).astype(float)
        # garganta: sector del anillo con x > w/L + R1 (a la derecha del agujero) y |y| < R1/2
        g = (d > 0) & (X[:, 0] > w / L) & (np.abs(X[:, 1]) < 0.5 * R1)
        fila = dict(w=w)
        for n in range(4):
            r = rho[:, n]
            norma = uno @ (A @ r)
            fila[f"x_medio_nm_{n+1}"] = float(L * (X[:, 0] @ (A @ r)) / norma)
            fila[f"P_barrera_{n+1}"] = float(1 - d @ (A @ r) / norma)
            fila[f"P_garganta_{n+1}"] = float(g.astype(float) @ (A @ r) / norma)
        # ancho de la garganta y de la región ancha
        fila["d_min_nm"] = 28.0 - (w + 12.0)
        fila["d_max_nm"] = 28.0 - 12.0 + w
        filas.append(fila)
        print(f"  w={w:4.1f}  <x>_1={fila['x_medio_nm_1']:+6.2f} nm  P_barrera_1={fila['P_barrera_1']:.4f}  "
              f"P_garganta_1={fila['P_garganta_1']:.2e}", flush=True)
    res["P3"] = filas


# ======================================================================
# P4: figuras
# ======================================================================
def parte_P4():
    P1 = res["P1"]["filas"]
    w = np.array([f["w"] for f in P1])
    E = np.array([f["E_meV"] for f in P1])
    Ei = np.array([f["E_inf_meV"] for f in P1])
    par = np.array([f["paridad"] for f in P1])

    # --- espectro ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.0), sharey=False)
    for n in range(K):
        ax[0].plot(w, E[:, n], "-", color="0.75", lw=0.8, zorder=1)
        sym = np.where(par[:, n] > 0, "o", "s")
        for wi, ei, s_, p_ in zip(w, E[:, n], sym, par[:, n]):
            ax[0].plot(wi, ei, s_, ms=4, color="C0" if p_ > 0 else "C3", zorder=2)
    ax[0].plot([], [], "o", color="C0", label="par en $y$ ($s$, $p_x$, $d_{x^2-y^2}$)")
    ax[0].plot([], [], "s", color="C3", label="impar en $y$ ($p_y$, $d_{xy}$)")
    ax[0].set_xlabel("Excentricidad $w$ (nm)"); ax[0].set_ylabel("$E_n$ (meV)")
    ax[0].set_title("(a) Barrera finita ($V_0=262$ meV), 10 niveles", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax[0].grid(alpha=0.3)
    for n in range(4):
        ax[1].plot(w, E[:, n], "-", color=f"C{n}", label=f"$n={n+1}$ finita")
        ax[1].plot(w, Ei[:, n], "--", color=f"C{n}")
    ax[1].plot([], [], "k--", label="pozo infinito")
    ax[1].set_xlabel("Excentricidad $w$ (nm)"); ax[1].set_ylabel("$E_n$ (meV)")
    ax[1].set_title("(b) Barrera finita (—) frente a pozo infinito (- -)", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figuras_fase5/espectro_vs_w.png", dpi=200); plt.close()

    # --- desdoblamiento ---
    P2 = res["P2"]
    w2 = np.array([f["w"] for f in P2["filas"]])
    d1 = np.abs([f["d1_meV"] for f in P2["filas"]])
    d2 = np.abs([f["d2_meV"] for f in P2["filas"]])
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    s_ = w2 > 0
    ax.loglog(w2[s_], d1[s_], "o-", label=r"doblete $|m|=1$ ($p_x$/$p_y$)")
    ax.loglog(w2[s_], d2[s_], "s-", label=r"doblete $|m|=2$ ($d$)")
    ref = np.array([0.25, 2.0])
    i1 = np.where(w2 == 1.0)[0][0]
    ax.loglog(ref, d1[i1] * (ref / 1.0) ** 2, "k:", lw=1, label=r"$\propto w^2$")
    ax.loglog(ref, d2[i1] * (ref / 1.0) ** 4, "k--", lw=1, label=r"$\propto w^4$")
    ax.axhline(P2["piso_meV"][0], color="C0", lw=0.6, alpha=0.6)
    ax.axhline(P2["piso_meV"][1], color="C1", lw=0.6, alpha=0.6)
    ax.set_xlabel("Excentricidad $w$ (nm)"); ax.set_ylabel(r"$|\Delta E|$ (meV)")
    ax.set_title("Ruptura de la degeneración (líneas: piso numérico a $w=0$)", fontsize=9)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figuras_fase5/desdoblamiento.png", dpi=200); plt.close()

    # --- localización ---
    P3 = res["P3"]
    w3 = np.array([f["w"] for f in P3])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for n in range(4):
        ax[0].plot(w3, [f[f"x_medio_nm_{n+1}"] for f in P3], "o-", ms=3, label=f"$n={n+1}$")
        ax[1].plot(w3, [100 * f[f"P_barrera_{n+1}"] for f in P3], "o-", ms=3, label=f"$n={n+1}$")
        ax[2].semilogy(w3, [max(f[f"P_garganta_{n+1}"], 1e-12) for f in P3], "o-", ms=3, label=f"$n={n+1}$")
    ax[0].set_ylabel(r"$\langle x\rangle$ (nm)"); ax[0].set_title("(a) Centroide de $|\\psi_n|^2$", fontsize=10)
    ax[1].set_ylabel("Probabilidad en la barrera (%)"); ax[1].set_title("(b) Penetración en AlGaAs", fontsize=10)
    ax[2].set_ylabel("Probabilidad en la garganta"); ax[2].set_title("(c) Ocupación de la garganta", fontsize=10)
    for a in ax:
        a.set_xlabel("$w$ (nm)"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figuras_fase5/localizacion.png", dpi=200); plt.close()

    # --- densidades para w = 0, 5, 10 ---
    fig, ax = plt.subplots(3, 6, figsize=(15, 7.6))
    t = np.linspace(0, 2 * np.pi, 300)
    for fila_ax, w_ in zip(ax, (0.0, 5.0, 10.0)):
        V, C = problema(w_)
        m, s, E_, Psi, _ = adaptativo(V, C, n_max=4.0e4)
        Psi, par_ = paridades(s, E_, Psi)
        for n, a in enumerate(fila_ax):
            d = Psi[:, n] ** 2
            a.tricontourf(L * m.nodos[:, 0], L * m.nodos[:, 1], m.elems, d / d.max(), 25, cmap="viridis")
            a.plot(L * np.cos(t), L * np.sin(t), "w-", lw=0.6)
            a.plot(w_ + 12 * np.cos(t), 12 * np.sin(t), "w-", lw=0.6)
            a.set_xlim(-32, 32); a.set_ylim(-32, 32); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"$w={w_:g}$, $n={n+1}$: {MEV*E_[n]:.2f} meV ({'+' if par_[n] > 0 else '−'})", fontsize=8)
    plt.tight_layout(); plt.savefig("figuras_fase5/densidades.png", dpi=170); plt.close()
    print("figuras listas")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["P1", "P1c", "P2", "P3", "P4"]):
        {"P1": parte_P1, "P1c": parte_P1c, "P2": parte_P2, "P3": parte_P3, "P4": parte_P4}[p]()
        json.dump(res, open(ARCH, "w"), indent=1)
    print("Listo.")
