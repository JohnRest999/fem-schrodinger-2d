import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # módulos del solver
import os, json, numpy as np, matplotlib
os.makedirs("figuras", exist_ok=True)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":10})
B=json.load(open("figuras_barrera/resultados_barrera.json"))["E2"]
A=json.load(open("figuras_adaptativo/resultados_adaptativo.json"))["A2"]
fig,ax=plt.subplots(1,2,figsize=(9.6,4.0))
a=ax[0]
for k,st,lab in [("ajustada_nodal","o--C2","ajustada, $V$ nodal"),
                 ("no_ajustada_nodal","s--C3","no ajustada, $V$ nodal"),
                 ("ajustada_cuadratura","o-C2","ajustada, cuadratura"),
                 ("no_ajustada_cuadratura_aritmetica","x:C7",r"no ajustada, cuadr., $\bar C_e$ aritmética"),
                 ("no_ajustada_cuadratura","s-C3",r"no ajustada, cuadr., $\bar C_e$ armónica")]:
    f=B[k]; a.loglog([r["nodos"] for r in f],[r["err_medio"] for r in f],st[:-2],color=st[-2:],label=lab,ms=4)
N=np.array([1.4e3,9.5e4]); a.loglog(N,1.2e-2*(N/N[0])**-1,"k:",lw=1,label="$N^{-1}$")
a.set_title("(a) Disco, $R=10$ nm: malla uniforme",fontsize=10)
a=ax[1]
for k,st,lab in [("U-A","o--C2","uniforme ajustada"),("U-N","s--C3","uniforme no ajustada"),
                 ("AD-N","^-C0","adaptativa no ajustada"),("AD-A","v-C1","adaptativa ajustada")]:
    f=A[k]; a.loglog([r["nodos"] for r in f],[r["err_medio"] for r in f],st[:-2],color=st[-2:],label=lab,ms=4)
a.axhline(1e-3,color="0.5",lw=0.6,ls="-.")
a.set_title("(b) Anillo excéntrico, $x_0=10$ nm",fontsize=10)
for a in ax:
    a.set_xlabel("Número de nodos $N$"); a.set_ylabel("Error relativo medio"); a.grid(alpha=.3,which="both"); a.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig("figuras/fig_convergencia_v3.pdf"); fig.savefig("figuras/fig_convergencia_v3.png",dpi=150)
