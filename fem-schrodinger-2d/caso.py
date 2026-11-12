"""
caso.py
=======
Resuelve un problema arbitrario de la ecuación de Schrödinger 2D con masa
efectiva variable a partir de ARCHIVOS DE ENTRADA:

    python caso.py casos/anillo_excentrico.json

El archivo JSON describe (longitudes en nm, energías en eV):

  "unidades":   {"L_nm": 28, "m_ref": 0.067}          escala de las unidades naturales
  "dominio":    {"forma": FORMA, "etiquetas": [...]}   o
                {"malla": {"nodos": f, "elementos": f, "aristas": f}}  (malla externa)
  "materiales": [{"nombre", "region": REGION | "resto", "V_eV": número | "expresión en x,y",
                  "m_rel": número}, ...]   (el primero que contiene al punto manda)
  "frontera":   {"etiqueta": ["dirichlet", 0] | ["neumann", 0] | ["robin", alpha, 0]}
  "malla":      {"h_nm": 1.0, "ajustada": true}
  "solver":     {"niveles": 8, "integracion": "cuadratura"}
  "adaptativo": {"activar": true, "theta": 0.5, "nodos_max": 60000}
  "salida":     {"carpeta": "resultados/caso", "graficas": true}

FORMA:  {"tipo": "circulo", "centro": [x, y], "R": r}
        {"tipo": "poligono", "vertices": [[x, y], ...]}      (antihorario)
        {"tipo": "rectangulo", "esquinas": [[x0, y0], [x1, y1]]}
REGION: {"incluir": [FORMA, ...], "excluir": [FORMA, ...]}

Para un dominio poligonal, "etiquetas" asigna una etiqueta a cada lado (en el
orden de los vértices); sin ella todos los lados son "borde". Un círculo tiene
la etiqueta "borde". Archivos de malla externa: nodos (x y por línea, en nm),
elementos (i j k, base 0) y aristas de frontera (i j etiqueta).

Salidas: energias.csv, resumen.json y, si se piden, malla.png y estados.png.
"""
import json, os, sys, time
import numpy as np
import sympy as sp

from fem_schrodinger2d import FEMSchrodinger2D, Malla, energia_natural, x as SX, y as SY
from pruebas import _delaunay_dominio, _circulo, _interior_hex
from adaptativo import estimador, marcar_dorfler, refinar_nvb


# ======================================================================
# Formas geométricas (en nm) -> funciones en unidades naturales
# ======================================================================
class Forma:
    def __init__(self, d):
        t = d["tipo"]
        if t == "rectangulo":
            (x0, y0), (x1, y1) = d["esquinas"]
            d = {"tipo": "poligono", "vertices": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
            t = "poligono"
        self.tipo = t
        if t == "circulo":
            self.c = np.array(d["centro"], float)
            self.R = float(d["R"])
        elif t == "poligono":
            self.V = np.array(d["vertices"], float)
            a = 0.5 * np.sum(self.V[:, 0] * np.roll(self.V[:, 1], -1) - np.roll(self.V[:, 0], -1) * self.V[:, 1])
            if a < 0:
                self.V = self.V[::-1]
        else:
            raise ValueError(f"Forma desconocida: {t}")

    def dentro(self, x, y):
        if self.tipo == "circulo":
            return np.hypot(x - self.c[0], y - self.c[1]) < self.R
        from matplotlib.path import Path
        P = np.column_stack([np.ravel(x), np.ravel(y)])
        return Path(self.V).contains_points(P).reshape(np.shape(x))

    def distancia_borde(self, P):
        if self.tipo == "circulo":
            return np.abs(np.hypot(P[:, 0] - self.c[0], P[:, 1] - self.c[1]) - self.R)
        d = np.full(len(P), np.inf)
        for a, b in zip(self.V, np.roll(self.V, -1, axis=0)):
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
            d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
        return d

    def puntos_borde(self, h):
        if self.tipo == "circulo":
            return _circulo(self.R, h, tuple(self.c))
        pts = []
        for a, b in zip(self.V, np.roll(self.V, -1, axis=0)):
            n = max(1, int(round(np.linalg.norm(b - a) / h)))
            pts += [a + (b - a) * i / n for i in range(n)]
        return np.array(pts)

    def lado_mas_cercano(self, p):
        d = [abs((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0]) / np.linalg.norm(b - a)
             for a, b in zip(self.V, np.roll(self.V, -1, axis=0))]
        return int(np.argmin(d))

    def bbox(self):
        if self.tipo == "circulo":
            return self.c - self.R, self.c + self.R
        return self.V.min(0), self.V.max(0)


class Region:
    def __init__(self, d):
        self.incl = [Forma(f) for f in d.get("incluir", [])]
        self.excl = [Forma(f) for f in d.get("excluir", [])]

    def dentro(self, x, y):
        r = np.zeros(np.shape(x), bool)
        for f in self.incl:
            r |= f.dentro(x, y)
        for f in self.excl:
            r &= ~f.dentro(x, y)
        return r

    def formas(self):
        return self.incl + self.excl


# ======================================================================
# Problema
# ======================================================================
class Caso:
    def __init__(self, archivo):
        self.archivo = archivo
        self.base = os.path.dirname(os.path.abspath(archivo))
        self.d = json.load(open(archivo))
        u = self.d.get("unidades", {})
        self.L = float(u.get("L_nm", 10.0))
        self.m_ref = float(u.get("m_ref", 0.067))
        self.E0 = energia_natural(self.L, self.m_ref)          # eV
        self.materiales = []
        for mat in self.d["materiales"]:
            reg = None if mat.get("region", "resto") == "resto" else Region(mat["region"])
            self.materiales.append(dict(nombre=mat.get("nombre", "?"), region=reg,
                                        V=self._funcion_eV(mat.get("V_eV", 0.0)),
                                        mu=self.m_ref / float(mat["m_rel"])))
        if self.materiales[-1]["region"] is not None:
            raise ValueError("El último material debe tener \"region\": \"resto\".")

    def _funcion_eV(self, v):
        """Número o expresión (x, y en nm, resultado en eV) -> función de (x, y) naturales."""
        if isinstance(v, str):
            f = sp.lambdify((SX, SY), sp.sympify(v, locals={"x": SX, "y": SY}), "numpy")
            return lambda x, y: np.broadcast_to(f(x * self.L, y * self.L), np.shape(x)) / self.E0
        c = float(v) / self.E0
        return lambda x, y: np.full(np.shape(x), c)

    def _material(self, x, y):
        """Índice del material en cada punto (x, y naturales)."""
        idx = np.full(np.shape(x), len(self.materiales) - 1)
        asignado = np.zeros(np.shape(x), bool)
        for i, m in enumerate(self.materiales[:-1]):
            s = m["region"].dentro(x * self.L, y * self.L) & ~asignado
            idx[s] = i
            asignado |= s
        return idx

    def V(self, x, y):
        idx = self._material(x, y)
        out = np.zeros(np.shape(x))
        for i, m in enumerate(self.materiales):
            s = idx == i
            if s.any():
                out[s] = m["V"](np.asarray(x)[s], np.asarray(y)[s])
        return out

    def C(self, x, y):
        mu = np.array([m["mu"] for m in self.materiales])
        return mu[self._material(x, y)]

    # ------------------------------------------------------------------
    def malla(self):
        dom = self.d["dominio"]
        if "malla" in dom:
            return self._malla_externa(dom["malla"])
        F = Forma(dom["forma"])
        cfg = self.d.get("malla", {})
        h = float(cfg.get("h_nm", 1.0))
        ajust = bool(cfg.get("ajustada", True))
        lo, hi = F.bbox()
        g = _interior_hex(lo[0], hi[0], lo[1], hi[1], h)
        ok = F.dentro(g[:, 0], g[:, 1]) & (F.distancia_borde(g) > 0.6 * h)
        capas = [F.puntos_borde(h)]
        curvas = [(F.R / self.L, tuple(F.c / self.L))] if F.tipo == "circulo" else []
        if ajust:
            for m in self.materiales[:-1]:
                for f in m["region"].formas():
                    q = f.puntos_borde(h)
                    q = q[F.dentro(q[:, 0], q[:, 1]) & (F.distancia_borde(q) > 0.3 * h)]
                    capas.append(q)
                    ok &= f.distancia_borde(g) > 0.6 * h
                    if f.tipo == "circulo":
                        curvas.append((f.R / self.L, tuple(f.c / self.L)))
        pts = np.unique(np.round(np.vstack(capas + [g[ok]]), 9), axis=0) / self.L
        etiq = dom.get("etiquetas")
        if F.tipo == "poligono" and etiq:
            clasificar = lambda p: etiq[F.lado_mas_cercano(p * self.L)]
        else:
            clasificar = lambda p: "borde"
        m = _delaunay_dominio(pts, lambda x, y: F.dentro(x * self.L, y * self.L), clasificar)
        m.curvas = curvas
        return m

    def _malla_externa(self, cfg):
        f = lambda k: os.path.join(self.base, cfg[k])
        nodos = np.loadtxt(f("nodos"), ndmin=2) / self.L
        elems = np.loadtxt(f("elementos"), dtype=int, ndmin=2)
        filas = [l.split() for l in open(f("aristas")) if l.strip()]
        ar = np.array([[int(a), int(b)] for a, b, _ in filas])
        et = {tuple(sorted(p)): e for p, (_, _, e) in zip(ar.tolist(), filas)}
        mid = {tuple(np.round(0.5 * (nodos[a] + nodos[b]), 12)): e for (a, b), e in
               zip(ar, [r[2] for r in filas])}
        clasificar = lambda p: mid.get(tuple(np.round(p, 12)), "borde")
        m = Malla.desde_triangulos(nodos, elems, clasificar)
        m.curvas = []
        return m

    # ------------------------------------------------------------------
    def resolver(self, verbose=True):
        sol_cfg = self.d.get("solver", {})
        k = int(sol_cfg.get("niveles", 6))
        integ = sol_cfg.get("integracion", "cuadratura")
        bc = {e: tuple(v) for e, v in self.d.get("frontera", {"borde": ["dirichlet", 0.0]}).items()}
        ad = self.d.get("adaptativo", {})
        m = self.malla()
        historia = []
        t0 = time.time()
        while True:
            s = FEMSchrodinger2D(m, V=self.V, coef=self.C, bc=bc, integracion=integ)
            E, Psi = s.resolver_autovalores(k, sigma=-1e-2 + min(0.0, float(np.min(self.V(m.nodos[:, 0], m.nodos[:, 1])))))
            fila = dict(nodos=m.n, elementos=len(m.elems), E_meV=(1e3 * self.E0 * E).tolist())
            if ad.get("activar", False):
                eta2 = estimador(s, E, Psi)
                fila["eta"] = float(np.sqrt(eta2.sum()))
            historia.append(fila)
            if verbose:
                print(f"  N={m.n:7d}  E(meV)={np.round(1e3 * self.E0 * E[:min(k, 6)], 3)}", flush=True)
            if not ad.get("activar", False) or m.n >= float(ad.get("nodos_max", 5e4)):
                break
            m = refinar_nvb(m, marcar_dorfler(eta2, float(ad.get("theta", 0.5))))
        self.m, self.s, self.E, self.Psi, self.historia = m, s, E, Psi, historia
        self.t = time.time() - t0
        return E, Psi

    def guardar(self):
        sal = self.d.get("salida", {})
        carpeta = os.path.join(self.base, sal.get("carpeta", "resultados"))
        os.makedirs(carpeta, exist_ok=True)
        EmeV = 1e3 * self.E0 * self.E
        with open(os.path.join(carpeta, "energias.csv"), "w") as f:
            f.write("n,E_natural,E_meV\n")
            for i, (a, b) in enumerate(zip(self.E, EmeV)):
                f.write(f"{i + 1},{a:.10g},{b:.10g}\n")
        json.dump(dict(caso=self.d.get("nombre", os.path.basename(self.archivo)), L_nm=self.L,
                       E0_meV=1e3 * self.E0, nodos=self.m.n, elementos=len(self.m.elems),
                       tiempo_s=self.t, E_meV=EmeV.tolist(), historia=self.historia),
                  open(os.path.join(carpeta, "resumen.json"), "w"), indent=1)
        if sal.get("graficas", True):
            self._graficas(carpeta)
        return carpeta

    def _graficas(self, carpeta):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        X = self.m.nodos * self.L
        mat = self._material(self.m.nodos[:, 0], self.m.nodos[:, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.tripcolor(X[:, 0], X[:, 1], self.m.elems, mat, cmap="Pastel1", shading="flat", alpha=0.8)
        ax.triplot(X[:, 0], X[:, 1], self.m.elems, lw=0.2, color="k")
        ax.set_aspect("equal"); ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        ax.set_title(f"{self.m.n} nodos")
        plt.tight_layout(); plt.savefig(os.path.join(carpeta, "malla.png"), dpi=160); plt.close()
        n = min(6, len(self.E))
        fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
        for i, a in enumerate(np.atleast_1d(ax)):
            d = self.Psi[:, i] ** 2
            a.tricontourf(X[:, 0], X[:, 1], self.m.elems, d / d.max(), 25, cmap="viridis")
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"$n={i+1}$: {1e3*self.E0*self.E[i]:.2f} meV", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(carpeta, "estados.png"), dpi=160); plt.close()


if __name__ == "__main__":
    for arch in sys.argv[1:]:
        print(f"Caso: {arch}")
        c = Caso(arch)
        c.resolver()
        print(f"  resultados en {c.guardar()}  ({c.t:.1f} s)")
