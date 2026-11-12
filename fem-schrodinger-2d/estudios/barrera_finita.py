"""
barrera_finita.py  (Fase 3)
===========================
Verificación del solver con barrera finita y masa efectiva variable
(GaAs / Al0.3Ga0.7As, parámetros de Vinasco et al., Sci. Rep. 9, 1427, 2019),
en UNIDADES NATURALES:

    longitudes en unidades de L,   energías en unidades de E0 = hbar^2/(2 m_in L^2)
    -div( mu grad psi ) + v psi = e psi,   mu = m_in/m*,   v = V/E0

E1  Disco: verificación contra la ecuación de BenDaniel-Duke (solución exacta).
E2  Disco: convergencia uniforme, 4 variantes
        {malla ajustada, no ajustada} x {V y C nodal, por cuadratura}.
E3  Disco: sensibilidad al tamaño del dominio (Rc) y a la condición en r = Rc.
E4  Disco: sensibilidad a nsub_corte (precisión de la cuadratura en elementos cortados).
E5  Anillo concéntrico y excéntrico de Vinasco et al. (R1 = 12, R2 = 28 nm).

Salida: figuras_barrera/*.png y figuras_barrera/resultados_barrera.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jv, kv, jvp, kvp
from scipy.optimize import brentq

from fem_schrodinger2d import FEMSchrodinger2D, energia_natural
from pruebas import malla_heteroestructura

os.makedirs("figuras_barrera", exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ----------------------------------------------------------------------
# Parámetros físicos (Vinasco et al. 2019)
# ----------------------------------------------------------------------
V0_EV = 0.262
M_IN, M_OUT = 0.067, 0.093
MU_OUT = M_IN / M_OUT                       # mu en la barrera (mu = 1 en el pozo)
res = {"V0_eV": V0_EV, "m_in": M_IN, "m_out": M_OUT}


def problema(dentro, v0):
    """Devuelve (V, coef) en unidades naturales a partir del indicador del pozo."""
    V = lambda x, y: np.where(dentro(x, y), 0.0, v0)
    C = lambda x, y: np.where(dentro(x, y), 1.0, MU_OUT)
    return V, C


# ----------------------------------------------------------------------
# Solución exacta: disco de radio 1 (unidades de R) con barrera finita
# ----------------------------------------------------------------------
def exacta_disco_barrera(v0, k, mu_out=MU_OUT, mmax=12):
    """Raíces de  a J_m'(a) K_m(b) - mu_out b K_m'(b) J_m(a) = 0,
    a = sqrt(e), b = sqrt((v0 - e)/mu_out). Equivale a la continuidad de psi y
    de mu dpsi/dr en r = 1 (BenDaniel-Duke). Devuelve los k primeros niveles
    con su degeneración (m != 0 cuenta doble) y las etiquetas (m, n)."""
    niveles = []
    for m in range(mmax + 1):
        def f(e):
            a, b = np.sqrt(e), np.sqrt((v0 - e) / mu_out)
            return a * jvp(m, a) * kv(m, b) - mu_out * b * kvp(m, b) * jv(m, a)
        es = np.linspace(1e-9, v0 * (1 - 1e-9), 20000)
        fv = f(es)
        n = 0
        for i in np.where(np.sign(fv[:-1]) * np.sign(fv[1:]) < 0)[0]:
            n += 1
            r = brentq(f, es[i], es[i + 1], xtol=1e-14)
            niveles += [(r, m, n)] * (1 if m == 0 else 2)
    niveles.sort()
    return np.array([e for e, _, _ in niveles[:k]]), [(m, n) for _, m, n in niveles[:k]]


def resolver(malla, V, C, k, integracion, bc="dirichlet", **kw):
    t0 = time.time()
    s = FEMSchrodinger2D(malla, V=V, coef=C, bc={"borde": (bc, 0.0)},
                         integracion=integracion, **kw)
    t1 = time.time()
    E, Psi = s.resolver_autovalores(k, sigma=-1e-2)
    return E, Psi, s, t1 - t0, time.time() - t1


def main():
    # ======================================================================
    # Disco: R = 10 nm  ->  L = R, dominio Rc = 2 (20 nm)
    # ======================================================================
    R_NM = 10.0
    E0 = energia_natural(R_NM, M_IN)           # eV
    v0 = V0_EV / E0
    K_NIV = 8
    ex, etiq = exacta_disco_barrera(v0, K_NIV)
    dentro_disco = lambda x, y: np.hypot(x, y) < 1.0
    Vd, Cd = problema(dentro_disco, v0)
    res["disco"] = dict(R_nm=R_NM, E0_meV=1e3 * E0, v0=v0, mu_out=MU_OUT,
                        exacta_e=ex.tolist(), exacta_meV=(1e3 * E0 * ex).tolist(), etiquetas=etiq)
    print(f"E0 = {1e3*E0:.4f} meV, v0 = {v0:.3f}, mu_out = {MU_OUT:.4f}")
    print("Exactos (meV):", np.round(1e3 * E0 * ex, 3))

    # ---------------- E2: convergencia (incluye la malla de E1) ----------------
    HS = [0.1, 0.05, 0.025, 0.0125]            # en unidades de R (1, 0.5, 0.25, 0.125 nm)
    E2 = {}
    for ajust in (True, False):
        for integ in ("nodal", "cuadratura"):
            clave = f"{'ajustada' if ajust else 'no_ajustada'}_{integ}"
            filas = []
            for h in HS:
                m = malla_heteroestructura(2.0, h, [(1.0, (0.0, 0.0))], ajustada=ajust)
                E, _, s, te, tv = resolver(m, Vd, Cd, K_NIV, integ)
                err = np.abs(E - ex) / ex
                filas.append(dict(h=h, h_nm=h * R_NM, nodos=m.n, cortados=int(s.cortados.sum()),
                                  e=E.tolist(), err=err.tolist(), err_medio=float(err.mean()),
                                  t_ens=te, t_eig=tv))
                print(f"E2 {clave:24s} h={h*R_NM:5.3f} nm  N={m.n:6d}  err E1={err[0]:.2e}  "
                      f"medio={err.mean():.2e}  (ens {te:.2f}s, eig {tv:.1f}s)", flush=True)
            em = np.array([f["err_medio"] for f in filas])
            for f, o in zip(filas[1:], np.log2(em[:-1] / em[1:])):
                f["orden"] = float(o)
            E2[clave] = filas
    # misma malla no ajustada con la media ARITMÉTICA de C (versión anterior)
    filas = []
    for h in HS:
        m = malla_heteroestructura(2.0, h, [(1.0, (0.0, 0.0))], ajustada=False)
        E, _, s, te, tv = resolver(m, Vd, Cd, K_NIV, "cuadratura", media_C="aritmetica")
        err = np.abs(E - ex) / ex
        filas.append(dict(h=h, h_nm=h * R_NM, nodos=m.n, e=E.tolist(), err=err.tolist(),
                          err_medio=float(err.mean())))
        print(f"E2 no_ajustada_cuad_aritm    h={h*R_NM:5.3f} nm  N={m.n:6d}  medio={err.mean():.2e}", flush=True)
    em = np.array([f["err_medio"] for f in filas])
    for f, o in zip(filas[1:], np.log2(em[:-1] / em[1:])):
        f["orden"] = float(o)
    E2["no_ajustada_cuadratura_aritmetica"] = filas
    res["E2"] = E2

    # ---------------- E1: tabla de verificación ----------------
    f1 = E2["ajustada_cuadratura"][2]          # h = 0.25 nm
    res["E1"] = dict(h_nm=f1["h_nm"], nodos=f1["nodos"],
                     E_meV=(1e3 * E0 * np.array(f1["e"])).tolist(),
                     exacta_meV=(1e3 * E0 * ex).tolist(), err=f1["err"])
    print("\nE1 (ajustada, cuadratura, h = 0.25 nm):")
    for (mm, nn), a, b, e in zip(etiq, res["E1"]["E_meV"], res["E1"]["exacta_meV"], f1["err"]):
        print(f"   (m={mm}, n={nn})  E_FEM={a:9.4f}  E_exacta={b:9.4f} meV  err={e:.1e}")

    # ---------------- E3: truncamiento del dominio ----------------
    # En la MISMA malla, Dirichlet en r = Rc sube los niveles y Neumann los baja;
    # su diferencia acota el error por truncar el dominio (independiente de h).
    E3 = []
    for Rc in (1.25, 1.5, 2.0):
        m = malla_heteroestructura(Rc, 0.05, [(1.0, (0.0, 0.0))], ajustada=True)
        ED = resolver(m, Vd, Cd, K_NIV, "cuadratura", bc="dirichlet")[0]
        EN = resolver(m, Vd, Cd, K_NIV, "cuadratura", bc="neumann")[0]
        d = np.abs(ED - EN) / ED
        E3.append(dict(Rc=Rc, nodos=m.n, dif_DN=d.tolist(), dif_DN_max=float(d.max())))
        print(f"E3 Rc={Rc}R  N={m.n:6d}  |E_D-E_N|/E: nivel 1 {d[0]:.1e}, máx (8 niveles) {d.max():.1e}")
    res["E3"] = E3

    # ---------------- E4: precisión de la cuadratura en elementos cortados ----------------
    E4 = []
    m = malla_heteroestructura(2.0, 0.025, [(1.0, (0.0, 0.0))], ajustada=False)
    for nc in (4, 8, 16, 24, 48):
        E, _, s, te, _ = resolver(m, Vd, Cd, K_NIV, "cuadratura", nsub_corte=nc)
        err = np.abs(E - ex) / ex
        E4.append(dict(nsub_corte=nc, err_medio=float(err.mean()), t_ens=te))
        print(f"E4 nsub_corte={nc:2d}  err medio={err.mean():.3e}  (ens {te:.2f}s)")
    res["E4"] = E4

    # ======================================================================
    # E5: anillos de Vinasco et al. (R1 = 12, R2 = 28 nm, Rc = 2 R2), L = R2
    # ======================================================================
    L5 = 28.0
    E0r = energia_natural(L5, M_IN)
    v0r = V0_EV / E0r
    r1 = 12.0 / L5
    E5 = {}
    for x0_nm in (0.0, 10.0):
        c = (x0_nm / L5, 0.0)
        dentro = lambda x, y, c=c: (np.hypot(x, y) < 1.0) & (np.hypot(x - c[0], y - c[1]) > r1)
        Vr, Cr = problema(dentro, v0r)
        filas = []
        for h_nm in (1.0, 0.7, 0.5, 0.35):
            m = malla_heteroestructura(2.0, h_nm / L5, [(1.0, (0, 0)), (r1, c)], ajustada=True)
            E, _, _, te, tv = resolver(m, Vr, Cr, 6, "cuadratura")
            filas.append(dict(h_nm=h_nm, nodos=m.n, E_meV=(1e3 * E0r * E).tolist()))
            print(f"E5 x0={x0_nm:4.1f} h={h_nm:4.2f} nm N={m.n:6d}  E(meV)={np.round(1e3*E0r*E[:4],3)}"
                  f"  (eig {tv:.1f}s)", flush=True)
        # extrapolación de Richardson con orden 2 en h (dos mallas más finas)
        h1, h2 = filas[-2]["h_nm"], filas[-1]["h_nm"]
        E1_, E2_ = np.array(filas[-2]["E_meV"]), np.array(filas[-1]["E_meV"])
        rich = E2_ + (E2_ - E1_) / ((h1 / h2) ** 2 - 1)
        E5[f"x0_{x0_nm:g}"] = dict(filas=filas, richardson_meV=rich.tolist())
        print(f"   Richardson: {np.round(rich[:4], 3)} meV")
    res["E5"] = E5
    res["E5"]["referencia_Vinasco_E0_B0_meV"] = 14.39

    # ======================================================================
    # FIGURA: convergencia E2
    # ======================================================================
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    estilos = {"ajustada_cuadratura": ("C2", "o-", "ajustada, cuadratura"),
               "ajustada_nodal": ("C2", "o--", "ajustada, nodal"),
               "no_ajustada_cuadratura": ("C3", "s-", "no ajustada, cuadratura"),
               "no_ajustada_nodal": ("C3", "s--", "no ajustada, nodal"),
               "no_ajustada_cuadratura_aritmetica": ("C7", "x:", "no ajustada, cuadratura, media aritm. de $C$")}
    for clave, (col, est, lab) in estilos.items():
        N = [f["nodos"] for f in E2[clave]]
        ax.loglog(N, [f["err_medio"] for f in E2[clave]], est, color=col, label=lab)
    N = np.array([f["nodos"] for f in E2["ajustada_cuadratura"]])
    y0 = E2["ajustada_cuadratura"][0]["err_medio"]
    ax.loglog(N, 0.5 * y0 * (N / N[0]) ** -1.0, "k:", lw=1, label=r"$N^{-1}\sim h^2$")
    ax.loglog(N, 2.0 * y0 * (N / N[0]) ** -0.5, "k-.", lw=1, label=r"$N^{-1/2}\sim h$")
    ax.set_xlabel("Número de nodos $N$")
    ax.set_ylabel("Error relativo medio (8 niveles)")
    ax.set_title("Disco GaAs/Al$_{0.3}$Ga$_{0.7}$As ($R=10$ nm), malla uniforme")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figuras_barrera/convergencia_barrera.png", dpi=200); plt.close()

    json.dump(res, open("figuras_barrera/resultados_barrera.json", "w"), indent=1)
    print("Listo.")


if __name__ == "__main__":
    main()
