# Preprocesamiento de Texto

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Pipeline de Preprocesamiento](#pipeline-de-preprocesamiento)
3. [Normalización](#normalización)
4. [Tokenización](#tokenización)
5. [Eliminación de Stopwords](#eliminación-de-stopwords)
6. [Stemming](#stemming)
7. [TF-IDF: Frecuencia de Término - Frecuencia Inversa de Documento](#tf-idf)
8. [Requerimientos y Consideraciones](#requerimientos-y-consideraciones)
9. [Ejemplo Completo](#ejemplo-completo)

---

## Introducción

El preprocesamiento es la etapa fundamental de cualquier pipeline de PLN. Su objetivo es transformar texto en bruto (raw text), que es inherentemente no estructurado y ruidoso, en una representación limpia, normalizada y tokenizada que pueda ser procesada por algoritmos de machine learning.

> **Principio clave**: *"Garbage in, garbage out"*. La calidad del preprocesamiento determina directamente el techo de rendimiento de los modelos posteriores.

---

## Pipeline de Preprocesamiento

El flujo de trabajo sigue un orden estricto donde cada paso depende del anterior:

```
Texto Crudo
    │
    ▼
┌─────────────┐
│ Normalización │  → minúsculas, eliminar puntuación, caracteres especiales
└─────────────┘
    │
    ▼
┌─────────────┐
│ Tokenización  │  → dividir en palabras/unidades
└─────────────┘
    │
    ▼
┌─────────────┐
│ Stopwords     │  → eliminar palabras funcionales (el, la, de, en...)
└─────────────┘
    │
    ▼
┌─────────────┐
│ Stemming      │  → reducir a raíz morfológica
└─────────────┘
    │
    ▼
Tokens Procesados → listos para vectorización (TF-IDF, BoW, etc.)
```

---

## Normalización

### Objetivo
Uniformizar el texto para eliminar variaciones superficiales que no aportan información semántica.

### Pasos Detallados

#### 1. Conversión a Minúsculas
Todas las letras se convierten a minúsculas para que "Gato", "GATO" y "gato" sean tratados como el mismo token.

#### 2. Eliminación de Caracteres No Alfabéticos
Se remueven números, signos de puntuación, símbolos y cualquier carácter que no sea una letra del alfabeto español (incluyendo tildes y la ñ).

**Expresión regular utilizada:**
```python
[^a-záéíóúüñ\s]
```

- `^` dentro de `[]` significa negación
- `a-z` : letras minúsculas del alfabeto inglés
- `áéíóúüñ` : caracteres especiales del español
- `\s` : espacios en blanco (preservados)

#### 3. Normalización de Espacios
Múltiples espacios consecutivos se reducen a uno solo, y se eliminan espacios al inicio y final.

**Expresión regular:**
```python
\s+   # uno o más espacios en blanco
```

### Fórmula Formal

Dado un texto de entrada $t$, la normalización $N(t)$ se define como:

$$
N(t) = \text{trim}(\text{replace}(\text{lower}(t), \text{regex}_{\text{no-alfa}}, \epsilon))
$$

Donde:
- $\text{lower}(t)$: función de conversión a minúsculas
- $\text{regex}_{\text{no-alfa}}$: patrón que coincide con caracteres no alfabéticos
- $\epsilon$: cadena vacía (eliminación)
- $\text{trim}$: eliminación de espacios iniciales y finales

---

## Tokenización

### Definición
Proceso de dividir una secuencia de caracteres en unidades significativas llamadas **tokens**. En este proyecto, se utiliza tokenización a nivel de palabra (word tokenization).

### Algoritmo

```
Entrada: texto normalizado (string)
Salida: lista de tokens (List[str])

1. Dividir la cadena por cada carácter de espacio en blanco
2. Filtrar tokens vacíos resultantes de espacios múltiples
3. Retornar la lista
```

### Ejemplo

| Texto Normalizado | Tokens |
|-------------------|--------|
| `el gato come pescado` | `["el", "gato", "come", "pescado"]` |
| `los estudiantes están aprendiendo` | `["los", "estudiantes", "están", "aprendiendo"]` |

### Consideraciones Lingüísticas

En español, la tokenización por espacios es generalmente efectiva porque:
- Las palabras están separadas por espacios
- No hay casos complejos de clíticos como en inglés ("don't" → "do" + "n't")
- Sin embargo, contracciones del español como "del" (de + el) y "al" (a + el) pueden requerir tratamiento especial dependiendo de la aplicación

---

## Eliminación de Stopwords

### Definición
Las **stopwords** (palabras vacías) son términos de alta frecuencia en un idioma que aportan poco valor semántico para la discriminación entre documentos. Son principalmente artículos, preposiciones, conjunciones y pronombres.

### Justificación Matemática

En modelos basados en frecuencia (como TF-IDF o Naive Bayes), las stopwords dominan el vocabulario por su alta frecuencia pero baja información discriminativa.

Dado un corpus con $N$ documentos, la probabilidad condicional de una stopword $s$ dada cualquier clase $C$ tiende a ser uniforme:

$$
P(s | C_1) \approx P(s | C_2) \approx \dots \approx P(s | C_k)
$$

Por tanto, su contribución al log-likelihood ratio es aproximadamente cero:

$$
\log \frac{P(s | C_i)}{P(s | C_j)} \approx 0
$$

### Lista de Stopwords (Español)

El conjunto de stopwords implementado incluye:

| Categoría | Ejemplos |
|-----------|----------|
| Artículos | el, la, los, las, un, una, unos, unas |
| Preposiciones | de, en, por, con, para, del, al |
| Conjunciones | y, que |
| Pronombres | se, lo, le, su |
| Verbos auxiliares | es, son, no |

> **Nota**: La lista de stopwords debe adaptarse al dominio. En análisis de sentimiento, negaciones como "no" pueden ser informativas y no deberían eliminarse.

---

## Stemming

### Definición
El **stemming** es un proceso heurístico que reduce las palabras a su **raíz** o **stem**, eliminando sufijos flexivos y derivativos. A diferencia de la lematización, no garantiza producir una palabra válida del idioma.

### Algoritmo de Stemming Simple para Español

El implementado es un algoritmo regla-based que recorta sufijos comunes:

```
Para cada token:
    1. Verificar si termina con alguno de los sufijos en orden de longitud descendente
    2. Si coincide y la raíz resultante tiene longitud >= 3:
        - Retornar la raíz truncada
    3. Si no coincide ningún sufijo:
        - Retornar el token original
```

### Sufijos Implementados

| Sufijo | Tipo | Ejemplo |
|--------|------|---------|
| `-iendo` | Gerundio | aprendiendo → aprend |
| `-ando` | Gerundio | cantando → cant |
| `-iones` | Plural derivativo | naciones → nac |
| `-cion` | Sustantivo abstracto | nación → nac |
| `-mente` | Adverbio | rápidamente → rápid |
| `-ados/-adas` | Participio pasado plural | amados → am |
| `-idos/-idas` | Participio pasado plural | vividos → viv |
| `-es` | Plural | luces → luc |
| `-s` | Plural | gatos → gato |

### Limitaciones

Este stemmer es simplificado. Para producción se recomienda:
- **Snowball Stemmer** (Porter para español)
- **Lematización** con bibliotecas como spaCy, que usa análisis morfológico completo

---

## TF-IDF

### Definición
**TF-IDF** (Term Frequency - Inverse Document Frequency) es una métrica de ponderación que evalúa la importancia de una palabra en un documento relativa a una colección de documentos (corpus).

### Intuición
- **TF (Frecuencia de Término)**: Una palabra que aparece muchas veces en un documento es probablemente importante para ese documento.
- **IDF (Frecuencia Inversa de Documento)**: Una palabra que aparece en muchos documentos del corpus es común y por tanto menos discriminativa.

### Fórmulas Matemáticas

#### 1. Term Frequency (TF)

Mide la frecuencia relativa de un término $t$ en un documento $d$:

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

Donde:
- $f_{t,d}$: número de ocurrencias del término $t$ en el documento $d$
- $\sum_{t' \in d} f_{t',d}$: número total de términos en el documento $d$

**Variantes de TF** (no implementadas aquí, pero comunes):

- **TF booleano**: $\text{TF}(t,d) = 1$ si $t \in d$, sino $0$
- **TF logarítmico**: $\text{TF}(t,d) = 1 + \log(f_{t,d})$ si $f_{t,d} > 0$, sino $0$
- **TF augmentado**: $\text{TF}(t,d) = 0.5 + 0.5 \cdot \frac{f_{t,d}}{\max_{t'} f_{t',d}}$

#### 2. Inverse Document Frequency (IDF)

Mide la rareza de un término en el corpus:

$$
\text{IDF}(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right) = \log \left( \frac{N}{df_t} \right)
$$

Donde:
- $N = |D|$: número total de documentos en el corpus
- $df_t = |\{d \in D : t \in d\}|$: número de documentos que contienen el término $t$ (document frequency)
- $\log$: logaritmo natural (base $e$)

**Variante suavizada** (común en implementaciones como scikit-learn):

$$
\text{IDF}(t, D) = \log \left( \frac{N + 1}{df_t + 1} \right) + 1
$$

#### 3. TF-IDF Ponderado

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

### Propiedades Matemáticas

1. **TF-IDF = 0**: Si el término no aparece en el documento ($\text{TF} = 0$)
2. **TF-IDF alto**: Término frecuente en el documento pero raro en el corpus
3. **TF-IDF bajo**: Término raro en el documento o muy común en el corpus
4. **IDF mínimo**: $\log(N/N) = 0$ cuando el término aparece en todos los documentos

### Representación Vectorial

Cada documento $d$ se representa como un vector en un espacio de dimensiones $|V|$ (tamaño del vocabulario):

$$
\vec{d} = (\text{TF-IDF}(t_1, d), \text{TF-IDF}(t_2, d), \dots, \text{TF-IDF}(t_{|V|}, d))
$$

### Ejemplo de Cálculo

**Corpus:**
- Doc 1: "el gato come pescado fresco"
- Doc 2: "el perro come carne roja"
- Doc 3: "el gato y el perro juegan juntos"

**Después de preprocesamiento:**
- Doc 1: `["gato", "come", "pescado", "fresco"]`
- Doc 2: `["perro", "come", "carne", "roja"]`
- Doc 3: `["gato", "perro", "juegan", "juntos"]`

**Cálculo de IDF para "gato":**

$$
\text{IDF}(\text{"gato"}) = \log \left( \frac{3}{2} \right) \approx 0.405
$$

**Cálculo de TF-IDF para "gato" en Doc 1:**

$$
\text{TF}(\text{"gato"}, \text{Doc 1}) = \frac{1}{4} = 0.25
$$

$$
\text{TF-IDF}(\text{"gato"}, \text{Doc 1}) = 0.25 \times 0.405 \approx 0.101
$$

---

## Requerimientos y Consideraciones

### Requerimientos de Entrada

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texto` | `str` | Cadena de texto en español |
| `corpus_raw` | `List[str]` | Lista de documentos (strings) para TF-IDF |

### Complejidad Computacional

| Operación | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Normalización | $O(|t|)$ | $O(|t|)$ |
| Tokenización | $O(|t|)$ | $O(n)$ tokens |
| Stopwords | $O(n \cdot |S|)$ | $O(n)$ |
| Stemming | $O(n \cdot k)$ | $O(n)$ |
| TF-IDF | $O(N \cdot \bar{n} + |V| \cdot N)$ | $O(N \cdot |V|)$ |

Donde:
- $|t|$: longitud del texto
- $n$: número de tokens
- $|S|$: tamaño del conjunto de stopwords
- $k$: número de sufijos de stemming
- $N$: número de documentos
- $\bar{n}$: longitud promedio de documento
- $|V|$: tamaño del vocabulario

### Decisiones de Diseño

1. **Stopwords personalizadas**: Se usó una lista manual para español en lugar de NLTK para mantener independencia de bibliotecas externas pesadas.

2. **Stemmer simplificado**: Prioriza la claridad pedagógica sobre la precisión morfológica.

3. **Normalización agresiva**: Se eliminan todos los caracteres no alfabéticos, lo que puede perder información en textos con números significativos (ej: "Windows 11" → "Windows").

---

## Ejemplo Completo

```python
texto = "Los estudiantes están aprendiendo programación en la universidad"

# Paso 1: Normalización
# "los estudiantes están aprendiendo programación en la universidad"

# Paso 2: Tokenización
# ["los", "estudiantes", "están", "aprendiendo", "programación", "en", "la", "universidad"]

# Paso 3: Eliminar stopwords
# ["estudiantes", "están", "aprendiendo", "programación", "universidad"]

# Paso 4: Stemming
# ["estudiant", "están", "aprend", "programación", "universidad"]
```

---

## Referencias

- Salton, G. & Buckley, C. "Term-weighting approaches in automatic text retrieval" (1988)
- Porter, M. F. "An algorithm for suffix stripping" (1980)
- Manning, C. D. et al. *Introduction to Information Retrieval*, Cap. 6
