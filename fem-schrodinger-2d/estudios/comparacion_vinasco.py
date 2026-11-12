"""
comparacion_vinasco.py
======================
Comparación cuantitativa con Vinasco et al., Sci. Rep. 9, 1427 (2019).
Las curvas de sus Figs. 3(a) y 4(a) son gráficos vectoriales: se extraen las
coordenadas exactas de los trazos del PDF (pdfplumber) y se calibran con las
marcas de los ejes. Luego se comparan con nuestros cálculos FEM.

Uso: python comparacion_vinasco.py digitalizar | fig3 | figura
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import sys, json, os
import numpy as np

PDF = os.environ.get("PDF_VINASCO", "s41598-018-38114-0.pdf")   # PDF de Vinasco et al. (2019)
ARCH = "figuras_fase5/comparacion_vinasco.json"
res = json.load(open(ARCH)) if os.path.exists(ARCH) else {}


def _curvas(pagina, X0, SX, Y0, SY, ytop, ybot, xmax):
    import pdfplumber
    p = pdfplumber.open(PDF).pages[pagina]
    out = []
    for c in p.curves:
        P = np.array(c["pts"])
        if P[:, 0].min() < X0 - 1 or P[:, 0].max() > xmax + 1:
            continue
        m = (P[:, 1] >= ytop) & (P[:, 1] <= ybot)
        if m.sum() < 3:
            continue
        x, E = (P[m, 0] - X0) / SX, (Y0 - P[m, 1]) / SY
        if np.ptp(x) > 2:      # descarta rótulos dibujados como trazos ("B = 0", "(a)", ...)
            o = np.argsort(x)
            out.append((x[o].tolist(), E[o].tolist()))
    return p, out


def digitalizar():
    # Fig. 3(a), pág. 6: R1 = 0 en x = 192.0, 20 nm en x = 343.8; E = 0 en y = 406.2, 20 meV cada 79.85 pt
    p, c3 = _curvas(5, 192.0, (343.8 - 192.0) / 20, 406.2, 79.85 / 20, 47.0, 406.2, 359.1)
    rojos = sorted({round((406.2 - np.mean(np.array(o["pts"])[:, 1])) / (79.85 / 20), 2)
                    for k in p.objects for o in p.objects[k]
                    if "pts" in o and str(o.get("non_stroking_color")) == "(1.0, 0.0, 0.0)"
                    and np.array(o["pts"])[:, 0].max() < 200})
    # Fig. 4(a), pág. 7: w = 0 en x = 187.3, 10 nm en x = 360.3; 80 meV en y = 50.3, 20 meV cada 46.13 pt
    _, c4 = _curvas(6, 187.3, (360.3 - 187.3) / 10, 50.3 + 80 * 46.13 / 20, 46.13 / 20, 50.3, 223.3, 360.3)
    res.update(fig3a_curvas=c3, fig3a_disco_analitico=rojos, fig4a_curvas=c4)
    print(f"Fig. 3(a): {len(c3)} curvas, {len(rojos)} puntos analíticos; Fig. 4(a): {len(c4)} curvas")


def muestrear(curvas, xs, k, x_borde):
    """Niveles ordenados de las curvas en cada x. Solo las curvas que llegan al borde
    derecho del eje (x_borde) se extienden hasta él con su último valor."""
    T = np.array([np.interp(xs, x, E, left=np.nan,
                            right=E[-1] if x[-1] > x_borde - 0.3 else np.nan) for x, E in curvas])
    return np.array([np.sort(T[:, i][~np.isnan(T[:, i])])[:k] for i in range(len(xs))])


def fig3():
    """Nuestros anillos concéntricos para R1 = 1..21 nm (malla ajustada, h = 0.35 nm)."""
    from fem_schrodinger2d import FEMSchrodinger2D, energia_natural
    from pruebas import malla_heteroestructura
    from barrera_finita import MU_OUT
    L = 28.0; E0 = energia_natural(L, 0.067); v0 = 0.262 / E0
    filas = res.get("fig3a_nuestro", [])
    hechos = {f["R1"] for f in filas}
    for R1 in np.arange(1.0, 22.0, 2.0):
        if R1 in hechos:
            continue
        r1 = R1 / L
        d = lambda x, y: (np.hypot(x, y) < 1) & (np.hypot(x, y) > r1)
        m = malla_heteroestructura(1.5, min(0.35, R1 / 6) / L, [(1.0, (0, 0)), (r1, (0, 0))], ajustada=True)
        s = FEMSchrodinger2D(m, V=lambda x, y: np.where(d(x, y), 0, v0),
                             coef=lambda x, y: np.where(d(x, y), 1, MU_OUT),
                             bc={"borde": ("dirichlet", 0)}, integracion="cuadratura")
        E = 1e3 * E0 * s.resolver_autovalores(10, sigma=-1e-2)[0]
        filas.append(dict(R1=float(R1), nodos=m.n, E_meV=E.tolist()))
        print(f"  R1={R1:4.1f} N={m.n}  E={np.round(E[:4], 2)}", flush=True)
        res["fig3a_nuestro"] = filas
        json.dump(res, open(ARCH, "w"))


def figura():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from barrera_finita import exacta_disco_barrera, MU_OUT
    from fem_schrodinger2d import energia_natural
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    # (a) Fig. 3(a)
    a = ax[0]
    for x, E in res["fig3a_curvas"]:
        a.plot(x, E, "-", color="0.6", lw=0.9)
    nos = res["fig3a_nuestro"]
    for f in nos:
        a.plot([f["R1"]] * 10, f["E_meV"], "o", color="C0", ms=3.5, mfc="none")
    a.plot([0] * len(res["fig3a_disco_analitico"]), res["fig3a_disco_analitico"], "o", color="C3", ms=5,
           label="Vinasco: disco analítico (sus puntos rojos)")
    L = 28.0; E0 = energia_natural(L, 0.067)
    ex, _ = exacta_disco_barrera(0.262 / E0, 16)
    a.plot(np.zeros_like(ex) - 0.35, 1e3 * E0 * ex, "x", color="k", ms=6, label="nuestro disco analítico (BenDaniel–Duke)")
    a.plot([], [], "-", color="0.6", label="Vinasco: FEM (curvas digitalizadas)")
    a.plot([], [], "o", color="C0", mfc="none", label="nuestro FEM")
    a.set_xlim(-1, 22); a.set_ylim(0, 60); a.set_xlabel("$R_1$ (nm)"); a.set_ylabel("$E$ (meV)")
    a.set_title("(a) Anillo concéntrico, $R_2=28$ nm (su Fig. 3a)", fontsize=10)
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)
    # (b) Fig. 4(a)
    a = ax[1]
    for x, E in res["fig4a_curvas"]:
        a.plot(x, E, "-", color="0.6", lw=0.9)
    P1 = json.load(open("figuras_fase5/resultados_fase5.json"))["P1"]["filas"]
    for f in P1:
        a.plot([f["w"]] * 10, f["E_meV"], "o", color="C0", ms=3.5, mfc="none")
    a.plot([], [], "-", color="0.6", label="Vinasco: FEM (curvas digitalizadas)")
    a.plot([], [], "o", color="C0", mfc="none", label="nuestro FEM adaptativo")
    a.set_xlim(-0.3, 10.3); a.set_ylim(5, 50); a.set_xlabel("$w$ (nm)"); a.set_ylabel("$E$ (meV)")
    a.set_title("(b) Anillo excéntrico, $R_1=12$, $R_2=28$ nm, $B=0$ (su Fig. 4a)", fontsize=10)
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figuras_fase5/comparacion_vinasco.png", dpi=200); plt.close()
    # tabla de diferencias
    WS = np.array([f["w"] for f in P1]); Ev = muestrear(res["fig4a_curvas"], WS, 10, 10.0)
    En = np.array([f["E_meV"] for f in P1])
    d4 = np.abs(Ev - En)
    R1s = np.array([f["R1"] for f in nos]); E3v = muestrear(res["fig3a_curvas"], R1s, 10, 22.0)
    E3n = np.array([f["E_meV"] for f in nos]); d3 = np.abs(E3v - E3n)
    res["resumen"] = dict(fig4a_dif_media_meV=float(np.nanmean(d4)), fig4a_dif_max_meV=float(np.nanmax(d4)),
                          fig3a_dif_media_meV=float(np.nanmean(d3)), fig3a_dif_max_meV=float(np.nanmax(d3)),
                          fig3a_dif_rel_media=float(np.nanmean(d3 / E3v)))
    print(json.dumps(res["resumen"], indent=1))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        {"digitalizar": digitalizar, "fig3": fig3, "figura": figura}[p]()
        json.dump(res, open(ARCH, "w"))
