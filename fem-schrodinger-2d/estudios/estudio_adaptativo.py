"""
estudio_adaptativo.py  (Fase 4)
===============================
Comparación de estrategias de malla, en unidades naturales:

    U-A   uniforme, ajustada a las interfaces
    U-N   uniforme, no ajustada
    AD-N  adaptativa (estimador + Dörfler + NVB) desde una malla de fondo no ajustada
    AD-A  adaptativa desde una malla ajustada (nodos nuevos proyectados sobre las interfaces)

A1  Disco GaAs/Al0.3Ga0.7As, R = 10 nm: error frente a la solución exacta de
    BenDaniel-Duke (8 niveles), eficiencia del estimador y sensibilidad a theta.
A2  Anillo excéntrico de Vinasco et al. (R1 = 12, R2 = 28, x0 = 10 nm):
    error frente a una solución de referencia (AD-A muy fina).

Salida: figuras_adaptativo/*.png y figuras_adaptativo/resultados_adaptativo.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fem_schrodinger2d import FEMSchrodinger2D, energia_natural
from pruebas import malla_heteroestructura
from adaptativo import estimador, marcar_dorfler, refinar_nvb, calidad
from barrera_finita import exacta_disco_barrera, problema, V0_EV, M_IN

import sys
os.makedirs("figuras_adaptativo", exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
ARCH = "figuras_adaptativo/resultados_adaptativo.json"
res = json.load(open(ARCH)) if os.path.exists(ARCH) else {}
PARTES = sys.argv[1:] or ["A1", "A2", "figuras"]   # p. ej.: python estudio_adaptativo.py A2


def resolver(m, V, C, k):
    s = FEMSchrodinger2D(m, V=V, coef=C, bc={"borde": ("dirichlet", 0.0)},
                         integracion="cuadratura", nsub_corte=12)
    E, Psi = s.resolver_autovalores(k, sigma=-1e-2)
    return s, E, Psi


def ciclo_adaptativo(m, V, C, k, theta, N_max, ref=None, etiqueta=""):
    """Ejecuta RESOLVER-ESTIMAR-MARCAR-REFINAR hasta superar N_max nodos."""
    filas = []
    while True:
        t0 = time.time()
        s, E, Psi = resolver(m, V, C, k)
        eta2 = estimador(s, E, Psi)
        fila = dict(nodos=m.n, e=E.tolist(), eta=float(np.sqrt(eta2.sum())),
                    ang_min=calidad(m), t=time.time() - t0)
        if ref is not None:
            err = np.abs(E - ref) / ref
            fila.update(err=err.tolist(), err_medio=float(err.mean()))
        filas.append(fila)
        print(f"  {etiqueta} N={m.n:7d} " +
              (f"err medio={fila['err_medio']:.2e} " if ref is not None else "") +
              f"eta={fila['eta']:.2e} áng.mín={fila['ang_min']:.1f}°", flush=True)
        if m.n > N_max:
            return filas, m, s, E, Psi
        m = refinar_nvb(m, marcar_dorfler(eta2, theta))


def uniforme(hs, Rc, interfaces, ajustada, V, C, k, ref=None, etiqueta=""):
    filas = []
    for h in hs:
        m = malla_heteroestructura(Rc, h, interfaces, ajustada=ajustada)
        s, E, Psi = resolver(m, V, C, k)
        eta2 = estimador(s, E, Psi)
        fila = dict(h=h, nodos=m.n, e=E.tolist(), eta=float(np.sqrt(eta2.sum())))
        if ref is not None:
            err = np.abs(E - ref) / ref
            fila.update(err=err.tolist(), err_medio=float(err.mean()))
        filas.append(fila)
        print(f"  {etiqueta} h={h:.4f} N={m.n:7d} " +
              (f"err medio={fila['err_medio']:.2e} " if ref is not None else "") +
              f"eta={fila['eta']:.2e}", flush=True)
    return filas


def orden_final(filas, N_min=5e3):
    """Pendiente de log(err) vs log(N) ajustada por mínimos cuadrados sobre las
    mallas con N >= N_min (orden en N; el óptimo de P1 es 1). Se ajusta sobre
    varias mallas porque en las sucesiones adaptativas la pendiente entre dos
    mallas consecutivas fluctúa."""
    N = np.array([f["nodos"] for f in filas], float)
    e = np.array([f["err_medio"] for f in filas])
    s = N >= N_min
    return float(-np.polyfit(np.log(N[s]), np.log(e[s]), 1)[0])


# ======================================================================
# A1: disco (R = 10 nm, Rc = 2R), solución exacta
# ======================================================================
E0 = energia_natural(10.0, M_IN)
v0 = V0_EV / E0
K = 8
ex, _ = exacta_disco_barrera(v0, K)
Vd, Cd = problema(lambda x, y: np.hypot(x, y) < 1.0, v0)
interf = [(1.0, (0.0, 0.0))]
H0, NMAX, THETA = 0.2, 1.2e5, 0.5
L = 28.0
E0r = energia_natural(L, M_IN)
v0r = V0_EV / E0r
r1, c1 = 12.0 / L, (10.0 / L, 0.0)
dentro = lambda x, y: (np.hypot(x, y) < 1.0) & (np.hypot(x - c1[0], y - c1[1]) > r1)
Vr, Cr = problema(dentro, v0r)
interf_r = [(1.0, (0.0, 0.0)), (r1, c1)]
KR = 6
h0r = 2.0 / L


def parte_A1():
    print("A1: disco")
    A1 = {}
    A1["U-A"] = uniforme([0.1, 0.05, 0.025, 0.0125], 2.0, interf, True, Vd, Cd, K, ex, "U-A ")
    A1["U-N"] = uniforme([0.1, 0.05, 0.025, 0.0125], 2.0, interf, False, Vd, Cd, K, ex, "U-N ")
    A1["AD-N"], mN, sN, EN, PsiN = ciclo_adaptativo(
        malla_heteroestructura(2.0, H0, interf, ajustada=False), Vd, Cd, K, THETA, NMAX, ex, "AD-N")
    A1["AD-A"], mA, _, _, _ = ciclo_adaptativo(
        malla_heteroestructura(2.0, H0, interf, ajustada=True), Vd, Cd, K, THETA, NMAX, ex, "AD-A")
    for t in (0.3, 0.7):
        A1[f"AD-N_theta{t}"] = ciclo_adaptativo(
            malla_heteroestructura(2.0, H0, interf, ajustada=False), Vd, Cd, K, t, NMAX, ex,
            f"AD-N θ={t}")[0]
    res["A1"] = dict(E0_meV=1e3 * E0, exacta=ex.tolist(), theta=THETA, h0=H0, curvas=A1,
                     orden_en_N={k_: orden_final(v) for k_, v in A1.items()})
    for k_, o in res["A1"]["orden_en_N"].items():
        print(f"  orden en N (ajuste N >= 5e3) {k_:12s}: {o:.2f}   (óptimo P1: 1.00)")

    # eficiencia del estimador: err_medio / eta^2 debe mantenerse acotado
    for k_ in ("U-N", "AD-N"):
        ef = [f["err_medio"] / f["eta"] ** 2 for f in A1[k_]]
        print(f"  {k_}: err/eta^2 = " + ", ".join(f"{v:.3f}" for v in ef[-6:]))
        res["A1"][f"eficiencia_{k_}"] = ef


# ======================================================================
# A2: anillo excéntrico de Vinasco et al. (L = R2 = 28 nm, Rc = 2)
# ======================================================================
def parte_A2():
    print("A2: anillo excéntrico")
    # solución de referencia: AD-A llevada a ~2e5 nodos (límite de memoria) + extrapolación de Richardson
    # (orden 1 en N) con las dos últimas mallas
    ref_filas, _, _, _, _ = ciclo_adaptativo(
        malla_heteroestructura(2.0, h0r, interf_r, ajustada=True), Vr, Cr, KR, 0.5, 2.1e5, None, "ref ")
    Na, Nb = ref_filas[-2]["nodos"], ref_filas[-1]["nodos"]
    Ea, Eb = np.array(ref_filas[-2]["e"]), np.array(ref_filas[-1]["e"])
    ref_r = Eb + (Eb - Ea) / (Nb / Na - 1)
    print("  referencia (meV):", np.round(1e3 * E0r * ref_r, 4))
    A2 = {"referencia_meV": (1e3 * E0r * ref_r).tolist(),
          "referencia_sin_extrapolar_meV": (1e3 * E0r * Eb).tolist(), "N_referencia": Nb}
    A2["U-A"] = uniforme([1.0 / L, 0.7 / L, 0.5 / L, 0.35 / L], 2.0, interf_r, True, Vr, Cr, KR, ref_r, "U-A ")
    A2["U-N"] = uniforme([1.0 / L, 0.7 / L, 0.5 / L, 0.35 / L], 2.0, interf_r, False, Vr, Cr, KR, ref_r, "U-N ")
    A2["AD-N"], mRN, sRN, ERN, PsiRN = ciclo_adaptativo(
        malla_heteroestructura(2.0, h0r, interf_r, ajustada=False), Vr, Cr, KR, 0.5, 1.2e5, ref_r, "AD-N")
    A2["AD-A"] = ciclo_adaptativo(
        malla_heteroestructura(2.0, h0r, interf_r, ajustada=True), Vr, Cr, KR, 0.5, 1.2e5, ref_r, "AD-A")[0]
    res["A2"] = A2


# nodos necesarios para un error medio de 1e-3 (interpolación log-log)
def nodos_para(filas, tol=1e-3):
    N = np.array([f["nodos"] for f in filas], float)
    e = np.array([f["err_medio"] for f in filas])
    i = np.where(e <= tol)[0]
    if len(i) == 0:
        return None
    i = i[0]
    if i == 0:
        return float(N[0])
    s = np.log(e[i] / e[i - 1]) / np.log(N[i] / N[i - 1])
    return float(N[i - 1] * (tol / e[i - 1]) ** (1 / s))

def parte_figuras():
    for caso in ("A1", "A2"):
        curvas = res[caso]["curvas"] if caso == "A1" else res[caso]
        res[caso]["nodos_para_1e-3"] = {k_: nodos_para(curvas[k_]) for k_ in ("U-A", "U-N", "AD-N", "AD-A")}
        print(caso, "nodos para error medio 1e-3:",
              {k_: (round(v) if v else None) for k_, v in res[caso]["nodos_para_1e-3"].items()})

    # ======================================================================
    # FIGURAS
    # ======================================================================
    est = {"U-A": ("C2", "o--", "uniforme ajustada"), "U-N": ("C3", "s--", "uniforme no ajustada"),
           "AD-N": ("C0", "^-", "adaptativa no ajustada"), "AD-A": ("C1", "v-", "adaptativa ajustada")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    A1, A2 = res["A1"]["curvas"], res["A2"]
    for a, curvas, tit in ((ax[0], A1, "(a) Disco, $R=10$ nm (8 niveles, sol. exacta)"),
                           (ax[1], A2, "(b) Anillo excéntrico, $x_0=10$ nm (6 niveles)")):
        for k_, (col, st, lab) in est.items():
            a.loglog([f["nodos"] for f in curvas[k_]], [f["err_medio"] for f in curvas[k_]], st,
                     color=col, label=lab, ms=4)
        N = np.array([1e3, 2e5])
        e0 = curvas["U-A"][0]["err_medio"] * (curvas["U-A"][0]["nodos"] / N[0])
        a.loglog(N, 0.4 * e0 * (N / N[0]) ** -1, "k:", lw=1, label="$N^{-1}$")
        a.set_xlabel("Número de nodos $N$"); a.set_ylabel("Error relativo medio")
        a.set_title(tit, fontsize=10); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figuras_adaptativo/convergencia_adaptativa.png", dpi=200); plt.close()

    # mallas adaptadas y densidad del estado base
    def malla_y_densidad(m, Psi, nombre, lim, interfaces):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
        ax[0].triplot(m.nodos[:, 0], m.nodos[:, 1], m.elems, lw=0.15, color="k")
        ax[0].set_title(f"Malla adaptativa no ajustada ({m.n} nodos)", fontsize=10)
        d = Psi[:, 0] ** 2
        c = ax[1].tricontourf(m.nodos[:, 0], m.nodos[:, 1], m.elems, d / d.max(), 30, cmap="viridis")
        plt.colorbar(c, ax=ax[1], label=r"$|\psi_1|^2/\max$")
        ax[1].set_title("Densidad del estado base", fontsize=10)
        t = np.linspace(0, 2 * np.pi, 300)
        for a in ax:
            for R, cc in interfaces:
                a.plot(cc[0] + R * np.cos(t), cc[1] + R * np.sin(t), "r-", lw=0.8)
            a.set_aspect("equal"); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout(); plt.savefig(f"figuras_adaptativo/{nombre}.png", dpi=200); plt.close()

    # mallas de tamaño moderado para que se vean los elementos
    mD = malla_heteroestructura(2.0, H0, interf, ajustada=False)
    while mD.n < 8000:
        s, E, Psi = resolver(mD, Vd, Cd, K)
        mD = refinar_nvb(mD, marcar_dorfler(estimador(s, E, Psi), THETA))
    s, E, Psi = resolver(mD, Vd, Cd, K)
    malla_y_densidad(mD, Psi, "malla_disco", 2.0, interf)
    mR = malla_heteroestructura(2.0, h0r, interf_r, ajustada=False)
    while mR.n < 15000:
        s, E, Psi = resolver(mR, Vr, Cr, KR)
        mR = refinar_nvb(mR, marcar_dorfler(estimador(s, E, Psi), 0.5))
    s, E, Psi = resolver(mR, Vr, Cr, KR)
    malla_y_densidad(mR, Psi, "malla_anillo", 2.0, interf_r)


if __name__ == "__main__":
    for p in PARTES:
        {"A1": parte_A1, "A2": parte_A2, "figuras": parte_figuras}[p]()
        json.dump(res, open(ARCH, "w"), indent=1)
    print("Listo.")
