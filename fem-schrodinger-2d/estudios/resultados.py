"""
resultados.py
=============
Ejecuta las pruebas de validación con el solver (fem_schrodinger2d.py), usando
las mallas de pruebas.py, y guarda las figuras y tablas del informe de avance
en ./figuras/ y ./figuras/resultados.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jnp_zeros, jv, yv
from scipy.optimize import brentq
import sympy as sp

from fem_schrodinger2d import FEMSchrodinger2D
from fem_schrodinger2d import x as sx, y as sy
from pruebas import (HBAR, M_GAAS, J_M2_A_EV_NM2, malla_disco,
                     malla_anillo_asimetrico)

os.makedirs("figuras", exist_ok=True)
COEF = HBAR ** 2 / (2 * M_GAAS) * J_M2_A_EV_NM2          # eV nm^2
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
res = {"coef_eV_nm2": COEF}


def solver(malla, V, bc):
    return FEMSchrodinger2D(malla, V=V, hbar=HBAR, masa=M_GAAS, bc=bc,
                            factor_unidades=J_M2_A_EV_NM2)


# ------------------------------------------------------------------
# Soluciones analíticas
# ------------------------------------------------------------------
def exacta_disco(R, k, tipo="dirichlet"):
    out = []
    for m in range(0, 8):
        if tipo == "dirichlet":
            z = jn_zeros(m, 4)
            base = [0.0] if False else []
        else:
            z = jnp_zeros(m, 4)
            base = [0.0] if m == 0 else []
        zs = list(base) + list(z)
        mult = 1 if m == 0 else 2
        for zz in zs:
            out += [COEF * (zz / R) ** 2] * mult
    return np.sort(out)[:k]


def exacta_anillo_concentrico(Ri, Re, k):
    out = []
    kk = np.linspace(1e-3, 1.0, 20000)
    for m in range(0, 8):
        f = lambda q: jv(m, q * Ri) * yv(m, q * Re) - jv(m, q * Re) * yv(m, q * Ri)
        v = f(kk)
        for i in np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0][:4]:
            q = brentq(f, kk[i], kk[i + 1])
            out += [COEF * q ** 2] * (1 if m == 0 else 2)
    return np.sort(out)[:k]


# ------------------------------------------------------------------
# T1: disco, Dirichlet, convergencia en h
# ------------------------------------------------------------------
R = 10.0
hs = [1.6, 0.8, 0.4, 0.2]
T1 = []
for h in hs:
    mal = malla_disco(R, h)
    t0 = time.time()
    sol = solver(mal, 0.0, {"borde": ("dirichlet", 0.0)})
    t_ens = time.time() - t0
    t0 = time.time()
    E, Psi = sol.resolver_autovalores(8)
    t_eig = time.time() - t0
    ex = exacta_disco(R, 8)
    err = np.abs(E - ex) / ex
    T1.append(dict(h=h, nodos=mal.n, elems=len(mal.elems), E=E.tolist(),
                   exacta=ex.tolist(), err=err.tolist(),
                   t_ens=t_ens, t_eig=t_eig))
    print("T1", h, mal.n, err[0], err.mean())
res["T1"] = T1

# ------------------------------------------------------------------
# T2: disco, Neumann homogénea (BC natural)
# ------------------------------------------------------------------
mal = malla_disco(R, 0.4)
sol = solver(mal, 0.0, {"borde": ("neumann", 0.0)})
E, _ = sol.resolver_autovalores(7, sigma=-1e-3)
ex = exacta_disco(R, 7, "neumann")
res["T2"] = dict(h=0.4, nodos=mal.n, E=E.tolist(), exacta=ex.tolist())
print("T2", E, ex)

# ------------------------------------------------------------------
# T3: oscilador armónico (V != 0), disco grande con Dirichlet
# ------------------------------------------------------------------
hw = 0.020  # eV
V_ho = (hw ** 2) * (sx ** 2 + sy ** 2) / (4 * COEF)
Rho = 40.0
T3 = []
for h in [2.0, 1.0, 0.5]:
    mal = malla_disco(Rho, h)
    sol = solver(mal, V_ho, {"borde": ("dirichlet", 0.0)})
    E, _ = sol.resolver_autovalores(10)
    # niveles exactos: hw(2 n_r+|m|+1), degeneracia n+1 en N = 2n_r+|m|
    ex = np.sort(np.concatenate([[hw * (N + 1)] * (N + 1) for N in range(6)]))[:10]
    T3.append(dict(h=h, nodos=mal.n, E=E.tolist(), exacta=ex.tolist(),
                   err=(np.abs(E - ex) / ex).tolist()))
    print("T3", h, mal.n, np.abs(E - ex) / ex)
res["T3"] = T3

# ------------------------------------------------------------------
# T4: solución manufacturada, anillo excéntrico, BC mixtas
# ------------------------------------------------------------------
Re_, Ri_, x0_, y0_ = 10.0, 3.0, 2.0, 1.0
u = sp.cos(sx / 4) * sp.sin(sy / 5) + 1
Vm = 0.05 * (sx ** 2 + sy ** 2) + 1
f = -(u.diff(sx, 2) + u.diff(sy, 2)) + Vm * u
dun = u.diff(sx) * (-(sx - x0_) / Ri_) + u.diff(sy) * (-(sy - y0_) / Ri_)
bc = {"exterior": ("dirichlet", u), "interior": ("robin", 0.7, dun + 0.7 * u)}
T4 = []
for h in [2.0, 1.0, 0.5, 0.25, 0.125]:
    mal = malla_anillo_asimetrico(Re_, Ri_, x0_, y0_, h)
    s = FEMSchrodinger2D(mal, V=Vm, hbar=1.0, masa=0.5, bc=bc)
    uh = s.resolver_estacionario(0.0, f)
    T4.append(dict(h=h, nodos=mal.n, err=s.error_L2(uh, u)))
    print("T4", h, mal.n, T4[-1]["err"])
res["T4"] = T4

# ------------------------------------------------------------------
# T5: anillo concentrico (analítico) y anillo excéntrico
# ------------------------------------------------------------------
Re, Ri, x0 = 30.0, 10.0, 8.0
bcr = {"exterior": ("dirichlet", 0.0), "interior": ("dirichlet", 0.0)}
exc = exacta_anillo_concentrico(Ri, Re, 8)
T5 = []
for h in [3.0, 1.5, 0.75]:
    mal = malla_anillo_asimetrico(Re, Ri, 0.0, 0.0, h)
    E, _ = solver(mal, 0.0, bcr).resolver_autovalores(8)
    T5.append(dict(h=h, nodos=mal.n, E=E.tolist(), exacta=exc.tolist(),
                   err=(np.abs(E - exc) / exc).tolist()))
    print("T5", h, mal.n, np.abs(E - exc) / exc)
res["T5"] = T5

# Anillo excéntrico: convergencia uniforme + estados
T6 = []
for h in [3.0, 1.5, 0.75, 0.5]:
    mal = malla_anillo_asimetrico(Re, Ri, x0, 0.0, h)
    E, Psi = solver(mal, 0.0, bcr).resolver_autovalores(8)
    T6.append(dict(h=h, nodos=mal.n, E=E.tolist()))
    print("T6", h, mal.n, E[:4])
res["T6"] = T6
mal_exc = malla_anillo_asimetrico(Re, Ri, x0, 0.0, 0.75)
E_exc, Psi_exc = solver(mal_exc, 0.0, bcr).resolver_autovalores(8)
mal_con = malla_anillo_asimetrico(Re, Ri, 0.0, 0.0, 0.75)
E_con, Psi_con = solver(mal_con, 0.0, bcr).resolver_autovalores(8)
mal_dis = malla_disco(R, 0.4)
E_dis, Psi_dis = solver(mal_dis, 0.0, {"borde": ("dirichlet", 0.0)}).resolver_autovalores(8)
res["espectros"] = dict(excentrico=E_exc.tolist(), concentrico=E_con.tolist(),
                        h=0.75, nodos_exc=mal_exc.n)
# ancho de garganta y heurística de gradiente
res["garganta"] = dict(ancho_min=Re - (x0 + Ri), ancho_max=Re + x0 - (-Ri + 0) - Ri - Ri
                       if False else (Re + x0) - Ri + 0)

# ------------------------------------------------------------------
# FIGURAS
# ------------------------------------------------------------------
# F1 mallas
fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
m1 = malla_disco(R, 1.0)
ax[0].triplot(m1.nodos[:, 0], m1.nodos[:, 1], m1.elems, lw=0.5, color="k")
ax[0].set_title(f"Disco, $R=10$ nm ({m1.n} nodos)")
m2 = malla_anillo_asimetrico(Re, Ri, x0, 0.0, 1.5)
ax[1].triplot(m2.nodos[:, 0], m2.nodos[:, 1], m2.elems, lw=0.4, color="k")
ax[1].set_title(f"Anillo excéntrico ({m2.n} nodos)")
for a in ax:
    a.set_aspect("equal"); a.set_xlabel("x (nm)"); a.set_ylabel("y (nm)")
plt.tight_layout(); plt.savefig("figuras/mallas.png", dpi=200); plt.close()

# F2 convergencia Bessel
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
H = np.array([t["h"] for t in T1])
e1 = np.array([t["err"][0] for t in T1]); em = np.array([np.mean(t["err"]) for t in T1])
ax[0].loglog(H, e1, "o-", label="$E_1$"); ax[0].loglog(H, em, "s-", label="media 8 niveles")
ax[0].loglog(H, e1[0] * (H / H[0]) ** 2, "k--", lw=0.8, label="$O(h^2)$")
ax[0].set_xlabel("$h$ (nm)"); ax[0].set_ylabel("error relativo"); ax[0].legend()
ax[0].set_title("Disco, Dirichlet (Bessel)"); ax[0].grid(True, which="both", alpha=0.3)
H3 = np.array([t["h"] for t in T3])
e3 = np.array([np.mean(t["err"]) for t in T3])
ax[1].loglog(H3, e3, "o-", label="media 10 niveles")
ax[1].loglog(H3, e3[0] * (H3 / H3[0]) ** 2, "k--", lw=0.8, label="$O(h^2)$")
ax[1].set_xlabel("$h$ (nm)"); ax[1].legend(); ax[1].set_title("Oscilador armónico ($V\\neq0$)")
ax[1].grid(True, which="both", alpha=0.3)
H4 = np.array([t["h"] for t in T4]); e4 = np.array([t["err"] for t in T4])
ax[2].loglog(H4, e4, "o-", label="error $L^2$")
ax[2].loglog(H4, e4[0] * (H4 / H4[0]) ** 2, "k--", lw=0.8, label="$O(h^2)$")
ax[2].set_xlabel("$h$ (nm)"); ax[2].legend(); ax[2].set_title("Sol. manufacturada (BC mixtas)")
ax[2].grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.savefig("figuras/convergencia.png", dpi=200); plt.close()


def paneles(mal, Psi, E, nombre, filas=2, cols=3, extra=None):
    fig, ax = plt.subplots(filas, cols, figsize=(3.4 * cols, 3.2 * filas))
    for i, a in enumerate(ax.ravel()):
        d = Psi[:, i] ** 2
        c = a.tricontourf(mal.nodos[:, 0], mal.nodos[:, 1], mal.elems, d, 30, cmap="viridis")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"$n={i+1}$, $E={1000*E[i]:.2f}$ meV", fontsize=9)
    plt.tight_layout(); plt.savefig(f"figuras/{nombre}.png", dpi=200); plt.close()


paneles(mal_dis, Psi_dis, E_dis, "estados_disco")
paneles(mal_exc, Psi_exc, E_exc, "estados_anillo")

# F5 espectros
fig, ax = plt.subplots(figsize=(6.5, 4))
for j, (E, lab) in enumerate([(E_dis, "Disco"), (E_con, "Anillo concéntrico"),
                              (E_exc, "Anillo excéntrico")]):
    for e in E[:8]:
        ax.hlines(1000 * e, j - 0.3, j + 0.3, color=f"C{j}", lw=2)
ax.set_xticks(range(3)); ax.set_xticklabels(["Disco\n($R=10$ nm)", "Anillo concéntrico\n($x_0=0$)",
                                             "Anillo excéntrico\n($x_0=8$ nm)"])
ax.set_ylabel("$E$ (meV)"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figuras/espectros.png", dpi=200); plt.close()

# F6 densidad en la garganta para el primer estado del anillo excéntrico
fig, ax = plt.subplots(figsize=(5, 4.4))
d = Psi_exc[:, 0] ** 2
c = ax.tricontourf(mal_exc.nodos[:, 0], mal_exc.nodos[:, 1], mal_exc.elems, d, 30, cmap="magma")
plt.colorbar(c, ax=ax, label=r"$|\psi_1|^2$ (nm$^{-2}$)")
ax.set_aspect("equal"); ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
plt.tight_layout(); plt.savefig("figuras/densidad_E1_excentrico.png", dpi=200); plt.close()

json.dump(res, open("figuras/resultados.json", "w"), indent=1)
print("Listo.")
