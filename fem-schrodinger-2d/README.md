# fem-schrodinger-2d

Solucionador de elementos finitos (P1, Galerkin) para la ecuación de Schrödinger
estacionaria en 2D con **masa efectiva y potencial dependientes de la posición**,
con **refinamiento h-adaptativo**. Proyecto del curso FS3003 Métodos Numéricos
para EDP.

    -div( hbar^2/(2 m*(x,y)) grad psi ) + V(x,y) psi = E psi

Las condiciones de BenDaniel–Duke en las interfaces entre materiales se cumplen
de forma natural en la forma débil. Se admiten condiciones de Dirichlet, Neumann
y Robin por etiqueta de frontera.

## Instalación

```bash
pip install -r requirements.txt
```
Probado con Python 3.12, NumPy 2.5, SciPy 1.18, SymPy 1.14 y Matplotlib 3.11.

## Uso: resolver un problema desde un archivo de entrada

```bash
python caso.py casos/anillo_excentrico.json
```
Los resultados (`energias.csv`, `resumen.json`, `malla.png`, `estados.png`) se
escriben en la carpeta indicada en el archivo. Ejemplos incluidos:

| Archivo | Problema |
|---|---|
| `casos/disco_barrera.json` | Disco de GaAs en AlGaAs; malla no ajustada y adaptativa |
| `casos/anillo_excentrico.json` | Anillo excéntrico de Vinasco et al. (2019) |
| `casos/molecula_cuantica.json` | Dos puntos acoplados, dominio rectangular, Dirichlet + Neumann |
| `casos/oscilador_malla_externa.json` | Oscilador armónico con `V` como expresión y malla leída de archivos |

### Formato del archivo de entrada (longitudes en nm, energías en eV)

```json
{
  "unidades":   {"L_nm": 28, "m_ref": 0.067},
  "dominio":    {"forma": {"tipo": "circulo", "centro": [0, 0], "R": 56}},
  "materiales": [
    {"nombre": "GaAs", "region": {"incluir": [FORMA, ...], "excluir": [FORMA, ...]},
     "V_eV": 0.0, "m_rel": 0.067},
    {"nombre": "AlGaAs", "region": "resto", "V_eV": 0.262, "m_rel": 0.093}
  ],
  "frontera":   {"borde": ["dirichlet", 0.0]},
  "malla":      {"h_nm": 2.0, "ajustada": false},
  "solver":     {"niveles": 10, "integracion": "cuadratura"},
  "adaptativo": {"activar": true, "theta": 0.5, "nodos_max": 60000},
  "salida":     {"carpeta": "resultados/anillo", "graficas": true}
}
```
* `FORMA`: `{"tipo": "circulo", "centro": [x, y], "R": r}`,
  `{"tipo": "poligono", "vertices": [[x, y], ...]}` o
  `{"tipo": "rectangulo", "esquinas": [[x0, y0], [x1, y1]]}`.
* `V_eV` puede ser un número o una expresión en `x`, `y` (nm), p. ej. `"1.7585e-4*(x**2+y**2)"`.
* En un dominio poligonal, `"etiquetas"` da una etiqueta a cada lado (en el orden de los vértices).
* Malla externa: `"dominio": {"malla": {"nodos": f, "elementos": f, "aristas": f}}` con
  nodos `x y` (nm), elementos `i j k` (base 0) y aristas de frontera `i j etiqueta`.
* El primer material cuya región contiene al punto es el que se usa; el último debe ser `"resto"`.

## Estructura

| Archivo | Contenido |
|---|---|
| `fem_schrodinger2d.py` | Deducción simbólica de las matrices P1, ensamblaje vectorizado, integración nodal o por cuadratura (media armónica de la masa en elementos cortados), condiciones de frontera, `eigsh` |
| `adaptativo.py` | Estimador residual, marcado de Dörfler, bisección del vértice más reciente |
| `pruebas.py` | Generadores de mallas (disco, anillo, heteroestructura ajustada / no ajustada) y pruebas básicas |
| `caso.py` | Lectura de archivos de entrada y solución de problemas arbitrarios |
| `estudios/` | Scripts que generan las figuras y tablas del informe |

## Reproducir el informe

Ejecutar desde la raíz del repositorio (los resultados quedan en `figuras*/`):

| Script | Secciones / figuras del informe | Tiempo aprox. |
|---|---|---|
| `python fem_schrodinger2d.py` | prueba de fórmulas vectorizadas vs. sympy | segundos |
| `python estudios/resultados.py` | verificación (Tabla 2, Fig. 2) | ~10 s |
| `python estudios/barrera_finita.py` | disco con barrera finita, integración de V y C | ~2 min |
| `python estudios/estudio_adaptativo.py A1` / `A2` / `figuras` | refinamiento adaptativo (Tabla 3, Figs. 3–4) | varios min por parte |
| `python estudios/fig2_v3.py` | Fig. 3 | segundos |
| `python estudios/fase5_anillo.py P1 P1c P2 P3 P4` | física del anillo (Figs. 6–9) | P1 se ejecuta por partes (~4 min c/u) |
| `PDF_VINASCO=ruta.pdf python estudios/comparacion_vinasco.py digitalizar fig3 figura` | comparación con Vinasco et al. (Fig. 5, Tabla 4) | ~2 min |
| `python estudios/test_adaptativo.py` | conformidad y calidad de la bisección | segundos |

`comparacion_vinasco.py` necesita el PDF del artículo (Sci. Rep. 9, 1427, 2019, de
acceso abierto) para extraer las curvas de sus figuras.

## Referencia principal

J. A. Vinasco et al., *Electronic states in GaAs-(Al,Ga)As eccentric quantum
rings under nonresonant intense laser and magnetic fields*, Sci. Rep. **9**, 1427 (2019).

## Autores

John Brainer Restrepo Tobón, Samuel Bermúdez Echeverri, Ana Isabel Osorio Zuluaga.
