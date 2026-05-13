# Procesamiento de Lenguaje Natural - Proyecto de Estudio

## Introducción al Procesamiento de Lenguaje Natural

El **Procesamiento de Lenguaje Natural (PLN o NLP por sus siglas en inglés)** es una disciplina de la inteligencia artificial que se encarga de la interacción entre las computadoras y el lenguaje humano. Su objetivo es permitir que las máquinas comprendan, interpreten y generen lenguaje natural de manera útil.

> **Definición formal**: El PLN es el campo que estudia cómo hacer que las computadoras procesen y analicen grandes cantidades de datos en lenguaje natural, transformando texto no estructurado en representaciones estructuradas que facilitan la extracción de información, la toma de decisiones y la comunicación máquina-humano.

### ¿Por qué es difícil?

El lenguaje humano es inherentemente **ambiguo**, **contextual** y **creativo**:

1. **Ambigüedad léxica**: "Banco" puede ser un asiento o una institución financiera
2. **Ambigüedad sintáctica**: "Vi al hombre con el telescopio" (¿quién tiene el telescopio?)
3. **Ambigüedad semántica**: "El pollo está listo para comer" (¿quién come?)
4. **Contexto cultural**: "Estar en la luna" no implica viaje espacial
5. **Evolución**: el lenguaje cambia constantemente (nuevas palabras, significados, jergas)

### Aplicaciones del PLN

| Área | Ejemplos |
|------|----------|
| **Búsqueda** | Google, motores de búsqueda semántica |
| **Asistentes virtuales** | Siri, Alexa, ChatGPT |
| **Traducción** | Google Translate, DeepL |
| **Análisis de sentimiento** | Reviews de productos, redes sociales |
| **Resumen automático** | TL;DR de artículos de noticias |
| **Chatbots** | Atención al cliente automatizada |
| **Corrección ortográfica** | Grammarly, autocorrector |
| **Extracción de información** | Análisis de contratos, currículums |

---

## Resumen de Temas del Proyecto

Este proyecto implementa un **pipeline completo de PLN** desde cero, cubriendo las etapas fundamentales desde el preprocesamiento hasta la búsqueda semántica. Cada módulo incluye implementación en Python (Jupyter Notebooks) y documentación teórica detallada con fórmulas matemáticas para preparación de exámenes.

### 1. Preprocesamiento de Texto
Transformación de texto crudo a tokens limpios y normalizados.

**Conceptos clave:** Normalización, tokenización, stopwords, stemming, TF-IDF.

**Fórmula principal:**

$$
\text{TF-IDF}(t, d, D) = \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF}} \times \underbrace{\log \left( \frac{N}{df_t} \right)}_{\text{IDF}}
$$

### 2. Modelos de Lenguaje N-gramas
Estimación de probabilidades de secuencias de palabras.

**Conceptos clave:** Regla de la cadena, hipótesis de Markov, bigramas, MLE, Laplace smoothing, perplejidad.

**Fórmulas principales:**

$$
P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

$$
P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

$$
\text{PP}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{i-1})\right)
$$

### 3. Clasificación con Naive Bayes
Asignación de categorías a documentos basada en probabilidades.

**Conceptos clave:** Teorema de Bayes, supuesto de independencia condicional, priors, likelihoods, suavizado, log-probabilidad.

**Fórmulas principales:**

$$
P(c | d) = \frac{P(d | c) \cdot P(c)}{P(d)}
$$

$$
\hat{c} = \arg\max_{c \in C} \left[ \log P(c) + \sum_{i=1}^{n} \log P(w_i | c) \right]
$$

$$
P_{\text{Laplace}}(w | c) = \frac{\text{conteo}(w, c) + 1}{|D_c| + |V|}
$$

### 4. Evaluación de Modelos
Métricas estadísticas para medir rendimiento de clasificadores.

**Conceptos clave:** Matriz de confusión, accuracy, precision, recall, F1-score, macro/micro average.

**Fórmulas principales:**

| Métrica | Fórmula |
|---------|---------|
| Accuracy | $\frac{VP + VN}{VP + VN + FP + FN}$ |
| Precision | $\frac{VP}{VP + FP}$ |
| Recall | $\frac{VP}{VP + FN}$ |
| F1 | $2 \cdot \frac{P \cdot R}{P + R}$ |

### 5. Word2Vec (Word Embeddings)
Representaciones vectoriales densas que capturan semántica.

**Conceptos clave:** Hipótesis distribucional, Skip-gram, CBOW, softmax, cross-entropy, similitud coseno, analogías.

**Fórmulas principales:**

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$$
\mathcal{L} = -\log P(w_t | w_c)
$$

$$
\text{similitud}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

### 6. Búsqueda Semántica
Recuperación de información basada en significado vectorial.

**Conceptos clave:** Embeddings de documentos, similitud vectorial, índices ANN (HNSW, IVF), bi-encoders, cross-encoders, re-ranking.

**Fórmulas principales:**

$$
\text{similitud}_{\cos}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}
$$

$$
\vec{d} = \frac{1}{|d|} \sum_{w \in d} \vec{e}_w
$$

$$
\text{score}_{\text{híbrido}} = \alpha \cdot \text{sim}_{\text{sem}} + (1 - \alpha) \cdot \text{score}_{\text{léc}}
$$

---

## Pipeline Completo de PLN

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1: PREPROCESAMIENTO                                           │
│  Texto crudo → Normalización → Tokenización → Stopwords → Stemming  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 2: REPRESENTACIÓN                                             │
│  • TF-IDF: vectores dispersos basados en frecuencia                 │
│  • Word2Vec: embeddings densos basados en contexto                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ FASE 3: MODELOS │       │ FASE 4: CLASIF. │       │ FASE 5: BÚSQUEDA│
│ DE LENGUAJE     │       │                 │       │ SEMÁNTICA       │
│                 │       │                 │       │                 │
│ • N-gramas      │       │ • Naive Bayes   │       │ • Embeddings    │
│ • Predicción    │       │ • Análisis de   │       │   de documentos │
│   de siguiente  │       │   sentimiento   │       │ • Índices ANN   │
│   palabra       │       │ • Spam detection│       │ • Ranking       │
│ • Generación    │       │ • Categorización│       │   vectorial     │
│   de texto      │       │   de noticias   │       │ • Re-ranking    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 6: EVALUACIÓN                                                 │
│  Matriz de confusión → Accuracy → Precision → Recall → F1-Score    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
.
├── preprocesamiento/
│   ├── preprocesamiento_basico.ipynb    # Normalización, tokenización, TF-IDF
│   └── README.md                        # Documentación teórica completa
│
├── n-gramas/
│   ├── bigramas.ipynb                   # Modelo de lenguaje bigramas
│   └── README.md                        # MLE, Laplace, perplejidad
│
├── bayes/
│   ├── naive_bayes.ipynb                # Clasificador Naive Bayes
│   └── README.md                        # Teorema de Bayes, log-prob
│
├── evaluacion/
│   ├── evaluadores.ipynb                # Métricas de clasificación
│   └── README.md                        # Matriz de confusión, F1, etc.
│
├── word2vec/
│   ├── word_2_vec_skip_gram.ipynb       # Embeddings con Skip-gram
│   └── README.md                        # Softmax, backprop, coseno
│
├── motor_busqueda_semantica/
│   ├── busqueda_semantica.ipynb         # Motor integrado: W2V + Naive Bayes + Búsqueda
│   └── README.md                        # Búsqueda vectorial, ANN, ranking
│
├── docs/
│   └── pipeline_nlp.md                  # Visión general del pipeline
│
├── pyproject.toml                       # Dependencias
└── README.md                            # Este archivo (índice central)
```

---

## Guía de Estudio por Módulos

Haz clic en cada enlace para acceder a la documentación completa con fórmulas y ejercicios.

| Módulo | Descripción | Documentación |
|--------|-------------|---------------|
| **Preprocesamiento** | Limpieza de texto, TF-IDF | [`preprocesamiento/README.md`](preprocesamiento/README.md) |
| **N-gramas** | Modelos de lenguaje, bigramas | [`n-gramas/README.md`](n-gramas/README.md) |
| **Naive Bayes** | Clasificación probabilística | [`bayes/README.md`](bayes/README.md) |
| **Evaluación** | Métricas de rendimiento | [`evaluacion/README.md`](evaluacion/README.md) |
| **Word2Vec** | Embeddings densos | [`word2vec/README.md`](word2vec/README.md) |
| **Búsqueda Semántica** | Recuperación vectorial | [`motor_busqueda_semantica/README.md`](motor_busqueda_semantica/README.md) |

---

## Requerimientos y Configuración

### Dependencias
- **Python**: >= 3.10
- **NumPy**: Álgebra lineal y operaciones numéricas
- **Jupyter Notebook**: Entorno interactivo de desarrollo

### Instalación

Este proyecto usa `uv` como gestor de paquetes:

```bash
# Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate

# Instalar dependencias
uv pip install -r pyproject.toml
```

### Ejecutar los Notebooks

```bash
jupyter notebook
```

Navega a cualquier carpeta y abre el archivo `.ipynb` correspondiente.

---

## Fórmulas Fundamentales del Pipeline

### Preprocesamiento

$$
\text{TF}(t,d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}} \quad \quad \text{IDF}(t,D) = \log \frac{N}{df_t}
$$

$$
\text{TF-IDF}(t,d,D) = \text{TF}(t,d) \times \text{IDF}(t,D)
$$

### N-gramas

$$
P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})} \quad \quad P_{\text{Laplace}} = \frac{C + 1}{C(w_{i-1}) + |V|}
$$

$$
\text{PP}(W) = \exp\left(-\frac{1}{N} \sum \log P(w_i | w_{i-1})\right)
$$

### Naive Bayes

$$
\hat{c} = \arg\max_c \left[ \log P(c) + \sum_{i=1}^{n} \log P(w_i | c) \right]
$$

$$
P(w | c) = \frac{\text{conteo}(w,c) + 1}{|D_c| + |V|}
$$

### Evaluación

$$
\text{Accuracy} = \frac{VP + VN}{\text{Total}} \quad \quad \text{Precision} = \frac{VP}{VP + FP}
$$

$$
\text{Recall} = \frac{VP}{VP + FN} \quad \quad \text{F1} = 2 \cdot \frac{P \cdot R}{P + R}
$$

### Word2Vec

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \quad \quad \mathcal{L} = -\log P(w_t | w_c)
$$

$$
\text{similitud}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

### Búsqueda Semántica

$$
\text{similitud}_{\cos}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} \quad \quad \vec{d} = \frac{1}{|d|} \sum_{w \in d} \vec{e}_w
$$

$$
\text{score}_{\text{híbrido}} = \alpha \cdot \text{sim}_{\text{sem}} + (1 - \alpha) \cdot \text{score}_{\text{léc}}
$$

---

## Checklist de Estudio para Examen

### Preprocesamiento
- [ ] ¿Puedo aplicar normalización paso a paso a un texto?
- [ ] ¿Entiendo la diferencia entre stemming y lematización?
- [ ] ¿Puedo calcular TF, IDF y TF-IDF manualmente?
- [ ] ¿Sé interpretar por qué una palabra tiene alto/bajo TF-IDF?

### N-gramas
- [ ] ¿Puedo escribir la regla de la cadena para una oración?
- [ ] ¿Entiendo la aproximación Markoviana?
- [ ] ¿Puedo calcular probabilidades MLE de bigramas?
- [ ] ¿Sé aplicar suavizado de Laplace?
- [ ] ¿Puedo calcular log-probabilidad de una oración?
- [ ] ¿Entiendo qué mide la perplejidad?

### Naive Bayes
- [ ] ¿Puedo enunciar el teorema de Bayes?
- [ ] ¿Entiendo el supuesto de independencia condicional?
- [ ] ¿Puedo calcular priors y likelihoods?
- [ ] ¿Sé por qué usamos logaritmos?
- [ ] ¿Puedo clasificar un documento paso a paso?
- [ ] ¿Entiendo el problema del cero y su solución?

### Evaluación
- [ ] ¿Puedo construir una matriz de confusión?
- [ ] ¿Sé identificar VP, VN, FP, FN?
- [ ] ¿Puedo calcular accuracy, precision, recall, F1?
- [ ] ¿Entiendo cuándo accuracy es engañosa?
- [ ] ¿Sé interpretar el trade-off precision-recall?

### Word2Vec
- [ ] ¿Entiendo la hipótesis distribucional?
- [ ] ¿Sé la diferencia entre Skip-gram y CBOW?
- [ ] ¿Puedo calcular softmax?
- [ ] ¿Entiendo por qué $W$ contiene los embeddings?
- [ ] ¿Puedo calcular similitud coseno?
- [ ] ¿Sé las limitaciones de Word2Vec vs BERT?

### Búsqueda Semántica
- [ ] ¿Entiendo la diferencia entre búsqueda léxica y semántica?
- [ ] ¿Puedo calcular similitud coseno entre vectores?
- [ ] ¿Sé cómo obtener un embedding de documento a partir de embeddings de palabra?
- [ ] ¿Entiendo por qué se usan índices ANN?
- [ ] ¿Sé la diferencia entre bi-encoder y cross-encoder?
- [ ] ¿Puedo explicar el pipeline de re-ranking?

---

## Formato de Datos por Módulo

| Módulo | Input Esperado | Output |
|--------|---------------|--------|
| **Preprocesamiento** | `str` (texto crudo) | `List[str]` (tokens procesados) |
| **TF-IDF** | `List[str]` (corpus) | `List[Dict]` (vectores TF-IDF) |
| **Bigramas** | `List[List[str]]` (oraciones tokenizadas) | Modelo probabilístico |
| **Naive Bayes** | `List[str]`, `List[str]` (docs, labels) | Clasificador entrenado |
| **Evaluación** | Modelo, `List[str]`, `List[str]` (test) | Métricas: acc, P, R, F1 |
| **Word2Vec** | `List[List[str]]` (corpus tokenizado) | Matriz de embeddings |
| **Búsqueda Semántica** | `List[str]` (documentos), `str` (query) | Top-k documentos similares |

---

## Referencias Generales

- Jurafsky, D. & Martin, J. H. *Speech and Language Processing* (3rd ed. draft)
- Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*
- Mikolov, T. et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
- McCallum, A. & Nigam, K. "A Comparison of Event Models for Naive Bayes Text Classification" (1998)
- Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
- Johnson, J. et al. "Billion-scale similarity search with GPUs" (2017)

---

## Autor

Proyecto educativo de Procesamiento de Lenguaje Natural.
