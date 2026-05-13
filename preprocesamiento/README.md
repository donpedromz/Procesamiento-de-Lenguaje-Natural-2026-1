# Preprocesamiento de Texto y TF-IDF

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Pipeline de Preprocesamiento](#pipeline-de-preprocesamiento)
3. [Normalización](#normalización)
4. [Tokenización](#tokenización)
5. [Eliminación de Stopwords](#eliminación-de-stopwords)
6. [Stemming](#stemming)
7. [TF-IDF (Term Frequency - Inverse Document Frequency)](#tf-idf)
8. [Resumen para Examen](#resumen-para-examen)
9. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

El **preprocesamiento** es la primera y más crítica etapa de cualquier pipeline de Procesamiento de Lenguaje Natural (PLN). Su propósito es transformar texto en bruto —que es inherentemente no estructurado, ruidoso y variable— en una representación limpia, normalizada y tokenizada que los algoritmos de machine learning puedan procesar.

> **Principio fundamental**: *Garbage in, garbage out* (GIGO). La calidad del preprocesamiento determina directamente el límite superior de rendimiento de cualquier modelo posterior.

### ¿Por qué es necesario?

- Las computadoras no entienden texto directamente; necesitan representaciones numéricas.
- El lenguaje humano es redundante: "correr", "corriendo", "corrió" comparten la misma raíz semántica.
- Palabras funcionales (artículos, preposiciones) dominan el conteo de frecuencias pero aportan poca información discriminativa.
- La puntuación y caracteres especiales suelen ser irrelevantes para tareas de clasificación.

---

## Pipeline de Preprocesamiento

El orden de aplicación es estricto y cada paso depende del anterior:

```
┌─────────────────────────────────────────────────────────────┐
│  TEXTO CRUDO                                                │
│  "Los estudiantes están aprendiendo programación!!!"         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  1. NORMALIZACIÓN                                           │
│  • Minúsculas                                               │
│  • Eliminar caracteres no alfabéticos                       │
│  • Normalizar espacios                                      │
│  Resultado: "los estudiantes están aprendiendo programación"│
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. TOKENIZACIÓN                                            │
│  • Dividir por espacios en blanco                           │
│  Resultado: ["los", "estudiantes", "están",                 │
│              "aprendiendo", "programación"]                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ELIMINACIÓN DE STOPWORDS                                │
│  • Filtrar palabras funcionales de alta frecuencia          │
│  Resultado: ["estudiantes", "están",                        │
│              "aprendiendo", "programación"]                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. STEMMING                                                │
│  • Reducir palabras a su raíz morfológica                   │
│  Resultado: ["estudiant", "están", "aprend", "programación"]│
└─────────────────────────────────────────────────────────────┘
```

---

## Normalización

### Definición
Proceso de uniformizar el texto eliminando variaciones superficiales que no aportan información semántica.

### Paso 1: Conversión a Minúsculas
**Razón**: En la mayoría de las tareas de PLN, la capitalización no es informativa. "Gato", "GATO" y "gato" deben tratarse como el mismo token.

$$
\text{lower}(t) : \Sigma^* \rightarrow \{a, b, \dots, z, á, é, \dots, ñ, \text{espacio}\}^*
$$

### Paso 2: Eliminación de Caracteres No Alfabéticos
Se remueven números, signos de puntuación, emojis y cualquier símbolo que no sea letra del español.

**Expresión Regular utilizada:**
```
[^a-záéíóúüñ\s]
```

| Componente | Significado |
|------------|-------------|
| `[^...]`   | Negación: coincide con cualquier carácter NO listado |
| `a-z`      | Letras minúsculas del alfabeto inglés |
| `áéíóúüñ`  | Caracteres especiales del español |
| `\s`       | Espacios en blanco (preservados) |

### Paso 3: Normalización de Espacios
Múltiples espacios consecutivos se reducen a uno, y se eliminan espacios iniciales y finales.

**Expresión regular:** `\s+` (uno o más espacios)

### Fórmula Formal
Dado un texto de entrada $t$, la normalización $N(t)$ se define como:

$$
N(t) = \text{trim}(\text{replace}(\text{lower}(t), \text{regex}_{\text{no-alfa}}, \epsilon))
$$

Donde:
- $\text{lower}(t)$: conversión a minúsculas
- $\text{regex}_{\text{no-alfa}}$: patrón de caracteres no alfabéticos
- $\epsilon$: cadena vacía
- $\text{trim}$: eliminación de espacios iniciales/finales

---

## Tokenización

### Definición
División de una secuencia de caracteres en unidades significativas llamadas **tokens**. En este proyecto se usa tokenización a nivel de palabra (*word tokenization*).

### Algoritmo
```
Entrada: texto normalizado (string)
Salida: lista de tokens (List[str])

1. Dividir la cadena por caracteres de espacio en blanco
2. Filtrar tokens vacíos
3. Retornar lista
```

### Ejemplo

| Texto Normalizado | Tokens Generados |
|-------------------|-----------------|
| `el gato come pescado` | `["el", "gato", "come", "pescado"]` |
| `los estudiantes están aprendiendo` | `["los", "estudiantes", "están", "aprendiendo"]` |

### Consideraciones Lingüísticas en Español
- Las palabras están separadas por espacios (a diferencia de idiomas como el chino o japonés)
- Las contracciones "del" (de + el) y "al" (a + el) pueden tratarse como tokens únicos o separarse según el caso de uso
- Los clíticos pronominales en español se escriben separados ("dime" = "di" + "me" en algunos análisis, pero tokenización simple lo deja como "dime")

---

## Eliminación de Stopwords

### Definición
Las **stopwords** (palabras vacías) son términos de muy alta frecuencia en un idioma que aportan poco valor semántico para la discriminación entre documentos.

### Justificación Matemática
En modelos basados en frecuencia (TF-IDF, Naive Bayes, etc.), las stopwords dominan el conteo pero su distribución es aproximadamente uniforme entre clases:

$$
P(s | C_1) \approx P(s | C_2) \approx \dots \approx P(s | C_k)
$$

Por tanto, su contribución al log-likelihood ratio es aproximadamente cero:

$$
\log \frac{P(s | C_i)}{P(s | C_j)} \approx 0
$$

### Lista de Stopwords Implementadas (Español)

| Categoría | Palabras |
|-----------|----------|
| Artículos | el, la, los, las, un, una, unos, unas |
| Preposiciones | de, en, por, con, para, del, al |
| Conjunciones | y, que |
| Pronombres | se, lo, le, su |
| Verbos comunes | es, son, no |

> **Advertencia para examen**: En análisis de sentimiento, negaciones como "no" pueden ser informativas. Eliminarlas sin criterio puede degradar el modelo.

---

## Stemming

### Definición
Proceso heurístico que reduce palabras a su **raíz** o **stem**, eliminando sufijos flexivos y derivativos. A diferencia de la **lemmatización**, no garantiza producir una palabra válida del idioma.

### Algoritmo Implementado
```
Para cada token:
    1. Verificar si termina con algún sufijo (en orden de longitud descendente)
    2. Si coincide Y la raíz resultante tiene longitud >= 3:
        → Retornar la raíz truncada
    3. Si no coincide ningún sufijo:
        → Retornar el token original
```

### Sufijos y Ejemplos

| Sufijo | Tipo Morfológico | Ejemplo |
|--------|-----------------|---------|
| `-iendo` | Gerundio | aprendiendo → aprend |
| `-ando` | Gerundio | cantando → cant |
| `-iones` | Plural derivativo | naciones → nac |
| `-cion` | Sustantivo abstracto | nación → nac |
| `-mente` | Adverbio | rápidamente → rápid |
| `-ados/-adas` | Participio pasado plural | amados → am |
| `-idos/-idas` | Participio pasado plural | vividos → viv |
| `-es` | Plural | luces → luc |
| `-s` | Plural | gatos → gato |

### Stemming vs. Lemmatización

| Característica | Stemming | Lemmatización |
|---------------|----------|---------------|
| Velocidad | Rápido | Más lento |
| Resultado | Puede no ser palabra válida | Siempre palabra válida |
| Método | Reglas heurísticas | Análisis morfológico completo |
| Ejemplo | "estudiantes" → "estudiant" | "estudiantes" → "estudiante" |
| Bibliotecas | Porter, Snowball | spaCy, NLTK WordNet |

---

## TF-IDF

### Definición
**TF-IDF** (Term Frequency - Inverse Document Frequency) es una métrica de ponderación que evalúa la importancia de un término dentro de un documento en relación con una colección de documentos (corpus).

> **Intuición**: Una palabra es importante para un documento si aparece frecuentemente en él (alto TF) pero raramente en el resto del corpus (alto IDF).

### 1. Term Frequency (TF)

Mide la frecuencia relativa de un término $t$ en un documento $d$:

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

Donde:
- $f_{t,d}$: número de ocurrencias del término $t$ en el documento $d$
- $\sum_{t' \in d} f_{t',d}$: número total de términos en el documento $d$

**Variantes comunes (para estudio):**

| Variante | Fórmula | Uso |
|----------|---------|-----|
| Booleana | $\text{TF}(t,d) = 1$ si $t \in d$, sino $0$ | Presencia/ausencia |
| Logarítmica | $1 + \log(f_{t,d})$ si $f_{t,d} > 0$, sino $0$ | Atenuar frecuencias altas |
| Augmentada | $0.5 + 0.5 \cdot \frac{f_{t,d}}{\max f_{t',d}}$ | Normalizar por máximo |

### 2. Inverse Document Frequency (IDF)

Mide la rareza de un término en todo el corpus:

$$
\text{IDF}(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right) = \log \left( \frac{N}{df_t} \right)
$$

Donde:
- $N = |D|$: número total de documentos
- $df_t = |\{d \in D : t \in d\}|$: número de documentos que contienen $t$ (*document frequency*)
- $\log$: logaritmo natural

**Variante suavizada (scikit-learn):**

$$
\text{IDF}(t, D) = \log \left( \frac{N + 1}{df_t + 1} \right) + 1
$$

### 3. TF-IDF Ponderado

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

### Propiedades Matemáticas Clave

| Situación | TF-IDF | Interpretación |
|-----------|--------|----------------|
| $t \notin d$ | $0$ | Término ausente |
| $t$ frecuente en $d$, raro en corpus | **Alto** | Término discriminativo |
| $t$ frecuente en $d$ y en corpus | **Bajo** | Término común (posible stopword) |
| $t$ en todos los documentos | $\text{IDF} = \log(N/N) = 0$ | Sin poder discriminativo |

### Representación Vectorial

Cada documento se representa como un vector en $\mathbb{R}^{|V|}$:

$$
\vec{d} = (\text{TF-IDF}(t_1, d), \text{TF-IDF}(t_2, d), \dots, \text{TF-IDF}(t_{|V|}, d))
$$

### Ejemplo de Cálculo Paso a Paso

**Corpus:**
- Doc 1: "el gato come pescado fresco"
- Doc 2: "el perro come carne roja"
- Doc 3: "el gato y el perro juegan juntos"

**Después de preprocesamiento:**
- Doc 1: `["gato", "come", "pescado", "fresco"]`
- Doc 2: `["perro", "come", "carne", "roja"]`
- Doc 3: `["gato", "perro", "juegan", "juntos"]`

**Pregunta de examen**: Calcular TF-IDF de "gato" en Doc 1.

**Solución:**

Paso 1: Calcular TF
$$
\text{TF}(\text{"gato"}, \text{Doc 1}) = \frac{1}{4} = 0.25
$$

Paso 2: Calcular IDF
- "gato" aparece en Doc 1 y Doc 3 → $df = 2$
- $N = 3$

$$
\text{IDF}(\text{"gato"}) = \log \left( \frac{3}{2} \right) \approx 0.405
$$

Paso 3: Calcular TF-IDF
$$
\text{TF-IDF}(\text{"gato"}, \text{Doc 1}) = 0.25 \times 0.405 \approx 0.101
$$

**Pregunta de examen**: Calcular TF-IDF de "come" en Doc 1.

- "come" aparece en Doc 1 y Doc 2 → $df = 2$
- TF = 1/4 = 0.25
- IDF = log(3/2) ≈ 0.405
- TF-IDF ≈ 0.101

> Nota: Aunque "gato" y "come" tienen el mismo TF-IDF en este caso, en corpus más grandes los nombres suelen tener IDF más alto que los verbos comunes.

---

## Resumen para Examen

### Preguntas Frecuentes de Examen

**1. ¿Por qué normalizamos a minúsculas?**
Para que "Casa", "CASA" y "casa" sean tratadas como el mismo token, evitando que el vocabulario se infle innecesariamente.

**2. ¿Cuál es la diferencia entre stemming y lemmatización?**
- Stemming: método heurístico rápido que puede generar raíces no válidas ("estudiantes" → "estudiant")
- Lemmatization: análisis morfológico completo que siempre genera palabras válidas ("estudiantes" → "estudiante")

**3. ¿Por qué eliminar stopwords?**
Porque son de alta frecuencia y baja información discriminativa. Su distribución es aproximadamente uniforme entre clases.

**4. ¿Qué problema resuelve TF-IDF que no resuelve el conteo simple (BoW)?**
TF-IDF penaliza términos que aparecen en muchos documentos (comunes, poco informativos) y resalta términos raros pero frecuentes en documentos específicos.

**5. ¿Por qué usamos logaritmo en IDF?**
Para comprimir el rango de valores. Sin logaritmo, un término que aparece en 1 de 1000 documentos tendría IDF = 1000, dominando desproporcionadamente los cálculos.

**6. ¿Qué pasa con palabras que aparecen en TODOS los documentos?**
IDF = log(N/N) = log(1) = 0. Por tanto, su TF-IDF es 0 independientemente de su frecuencia en el documento.

### Fórmulas a Memorizar

| Concepto | Fórmula |
|----------|---------|
| TF | $\text{TF}(t,d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$ |
| IDF | $\text{IDF}(t,D) = \log \frac{N}{df_t}$ |
| TF-IDF | $\text{TF-IDF} = \text{TF} \times \text{IDF}$ |

### Complejidad Computacional

| Operación | Complejidad |
|-----------|-------------|
| Normalización | $O(|t|)$ |
| Tokenización | $O(|t|)$ |
| Stopwords | $O(n \cdot |S|)$ |
| Stemming | $O(n \cdot k)$ |
| TF-IDF completo | $O(N \cdot \bar{n} + |V| \cdot N)$ |

Donde: $|t|$ = longitud texto, $n$ = tokens, $|S|$ = stopwords, $k$ = sufijos, $N$ = documentos, $\bar{n}$ = longitud promedio, $|V|$ = vocabulario.

---

## Ejercicios Propuestos

### Ejercicio 1: Normalización
Aplica el pipeline de preprocesamiento al siguiente texto:
> "¡Excelente servicio! Los meseros fueron muy atentos y la comida llegó rápido."

**Respuesta esperada:**
- Normalización: "excelente servicio los meseros fueron muy atentos y la comida llegó rápido"
- Tokens: `["excelente", "servicio", "los", "meseros", "fueron", "muy", "atentos", "y", "la", "comida", "llegó", "rápido"]`
- Sin stopwords: `["excelente", "servicio", "meseros", "fueron", "muy", "atentos", "comida", "llegó", "rápido"]`
- Stems: `["excelent", "servici", "meser", "fueron", "muy", "atent", "comid", "llegó", "rapid"]`

### Ejercicio 2: Cálculo de TF-IDF
Dado el corpus:
- D1: "el perro corre rápido"
- D2: "el gato corre lento"
- D3: "el perro y el gato corren juntos"

Calcula TF-IDF("corre", D1) y TF-IDF("perro", D1).

**Solución:**
- Preprocesamiento D1: `["perro", "corre", "rapid"]`
- TF("corre", D1) = 1/3
- "corre" aparece en D1 y D2 → df = 2, N = 3 → IDF = log(3/2) ≈ 0.405
- TF-IDF("corre", D1) ≈ 0.135

- TF("perro", D1) = 1/3
- "perro" aparece en D1 y D3 → df = 2 → IDF = log(3/2) ≈ 0.405
- TF-IDF("perro", D1) ≈ 0.135

### Ejercicio 3: Análisis Crítico
¿Por qué "llegó" no cambia en stemming en nuestra implementación?

**Respuesta:** Ninguno de los sufijos implementados (`iendo`, `ando`, `iones`, `cion`, `mente`, `ados`, `adas`, `idos`, `idas`, `es`, `s`) coincide con el final de "llegó". El sufijo "ó" de pretérito perfecto simple no está en la lista.

---

## Referencias para Estudio

- Salton, G. & Buckley, C. "Term-weighting approaches in automatic text retrieval" (1988)
- Porter, M. F. "An algorithm for suffix stripping" (1980)
- Manning, C. D. et al. *Introduction to Information Retrieval*, Capítulos 2 y 6
