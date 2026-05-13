# Búsqueda Semántica

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Búsqueda Léxica vs. Semántica](#búsqueda-léxica-vs-semántica)
3. [Pipeline de Búsqueda Semántica](#pipeline-de-búsqueda-semántica)
4. [Representación Vectorial de Documentos](#representación-vectorial-de-documentos)
5. [Métricas de Similitud y Distancia](#métricas-de-similitud-y-distancia)
6. [Indexación Vectorial](#indexación-vectorial)
7. [Recuperación y Ranking](#recuperación-y-ranking)
8. [Re-ranking y Refinamiento](#re-ranking-y-refinamiento)
9. [Requerimientos y Consideraciones](#requerimientos-y-consideraciones)
10. [Resumen para Examen](#resumen-para-examen)
11. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

La **búsqueda semántica** es una técnica de recuperación de información (Information Retrieval, IR) que busca comprender el **significado** detrás de las consultas y los documentos, en lugar de simplemente coincidir palabras clave.

> **Definición**: La búsqueda semántica mapea tanto documentos como consultas a un espacio vectorial continuo donde la proximidad refleja similitud semántica, permitiendo encontrar resultados conceptualmente relacionados incluso cuando no comparten términos exactos.

### Motivación

- La búsqueda por palabras clave no entiende sinónimos ("automóvil" vs "carro")
- No captura contexto ni intención del usuario
- No maneja polisemia (palabras con múltiples significados)
- La búsqueda semántica supera estas limitaciones mediante representaciones densas

---

## Implementación en este Proyecto

El notebook `busqueda_semantica.ipynb` implementa un **Motor de Búsqueda NLP integrado** que combina tres componentes fundamentales del pipeline de PLN desarrollados en los otros módulos:

### Arquitectura del MotorBusquedaNLP

```
┌─────────────────────────────────────────────────────────────┐
│               MotorBusquedaNLP                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Word2Vec    │    │ Naive Bayes │    │  Búsqueda   │     │
│  │ Skip-gram   │    │ Clasificador│    │  Vectorial  │     │
│  │ (dim=20)    │    │             │    │             │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                            ▼                                │
│              ┌─────────────────────────┐                    │
│              │  Indexación de Corpus   │                    │
│              │  • Entrena embeddings   │                    │
│              │  • Entrena clasificador │                    │
│              │  • Pre-calcula vectores │                    │
│              │    doc promedio         │                    │
│              └─────────────────────────┘                    │
│                            │                                │
│              ┌─────────────┴─────────────┐                  │
│              ▼                           ▼                  │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │  motor.buscar() │        │ motor.analizar()│            │
│  │  (query → top-k)│        │ (texto → clase  │            │
│  │                 │        │  + similares)   │            │
│  └─────────────────┘        └─────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Indexación (`indexar()`)

1. **Preprocesamiento**: Tokeniza todos los documentos del corpus
2. **Entrenamiento Word2Vec**: Aprende embeddings de palabras usando Skip-gram con los tokens del corpus
3. **Entrenamiento Naive Bayes** (opcional): Clasifica documentos en categorías predefinidas
4. **Vectorización de documentos**: Para cada documento, calcula el embedding promedio de sus palabras:

$$
\vec{d} = \frac{1}{|d|} \sum_{w \in d} \vec{e}_w
$$

### Flujo de Búsqueda (`buscar()`)

1. Preprocesa la consulta del usuario
2. Calcula el embedding promedio de la query
3. Compara con todos los vectores de documentos usando **similitud coseno**:

$$
\text{sim}(q, d_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\| \|\vec{d}_i\|}
$$

4. Retorna los top-$n$ documentos ordenados por similitud descendente

### Flujo de Análisis (`analizar()`)

1. Clasifica el texto usando Naive Bayes (obtiene categoría y scores)
2. Busca documentos similares usando búsqueda semántica
3. Retorna un diccionario con: categoría predicha, scores de clasificación, y documentos similares

### Ejemplo de Uso

```python
noticias = [
    'python se consolida como el lenguaje mas usado en ia',
    'las redes neuronales aprenden representaciones del lenguaje',
    'el procesamiento de lenguaje natural avanza con transformers',
]
etiquetas = ['tech', 'ia', 'ia']

motor = MotorBusquedaNLP()
motor.indexar(noticias, etiquetas)

# Búsqueda semántica
resultados = motor.buscar('redes neuronales y aprendizaje profundo', n=3)

# Análisis completo
analisis = motor.analizar('los transformers revolucionaron el pln')
# → {'sentimiento': 'ia', 'scores': {...}, 'documentos_similares': [...]}
```

> **Integración clave**: Este motor demuestra cómo los componentes individuales del pipeline (preprocesamiento, Word2Vec, Naive Bayes) se combinan para construir una aplicación de PLN funcional.

---

---

## Búsqueda Léxica vs. Semántica

### Búsqueda Léxica (Tradicional)

Basada en coincidencia exacta o parcial de términos:

- **TF-IDF matching**: $\text{score}(q, d) = \sum_{t \in q \cap d} \text{TF-IDF}(t, d)$
- **Boolean retrieval**: operadores AND, OR, NOT
- **BM25**: ranking probabilístico basado en frecuencias

**Limitaciones:**
- Sinónimos no se reconocen
- Requiere que el usuario use exactamente los mismos términos del documento
- No entiende relaciones conceptuales

### Búsqueda Semántica (Vectorial)

Basada en proximidad en espacio vectorial:

- Documentos y queries se representan como vectores densos
- La similitud se mide por distancia geométrica
- Captura relaciones semánticas implícitas aprendidas del corpus

| Característica | Léxica | Semántica |
|---------------|--------|-----------|
| Coincidencia | Términos exactos | Significado conceptual |
| Sinónimos | No reconoce | Sí reconoce |
| Polisemia | Confunde contextos | Diferencia por contexto |
| Escalabilidad | Inverted index eficiente | Requiere índice vectorial |
| Datos necesarios | Corpus para TF-IDF | Corpus grande para embeddings |

---

## Pipeline de Búsqueda Semántica

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 1: INDEXACIÓN                               │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│  Corpus Bruto   │───▶│ Preprocesamiento│───▶│ Embedding Model     │
│  (Documentos)   │    │ (tokenizar, etc)│    │ (Word2Vec, BERT,    │
└─────────────────┘    └─────────────────┘    │  Sentence-BERT, etc)│
                                              └─────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │  Vector Index       │
                                              │  (FAISS, Annoy,     │
                                              │   HNSW, etc.)       │
                                              └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 2: CONSULTA (QUERY)                         │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│  Query Usuario  │───▶│ Preprocesamiento│───▶│ Embedding Model     │
│  (Texto libre)  │    │                 │    │ (mismo modelo)      │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 3: RECUPERACIÓN                             │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│  Búsqueda en    │───▶│  Cálculo de     │───▶│  Ranking y         │
│  Índice Vectorial│    │  Similitud      │    │  Re-ranking        │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │  Top-k Resultados   │
                                              │  (más similares)    │
                                              └─────────────────────┘
```

---

## Representación Vectorial de Documentos

### De Palabras a Documentos

Existen múltiples estrategias para obtener un vector de documento a partir de vectores de palabra:

#### 1. Promedio de Embeddings (Pooling)

$$
\vec{d} = \frac{1}{|d|} \sum_{w \in d} \vec{e}_w
$$

Donde $\vec{e}_w$ es el embedding de la palabra $w$.

**Ventaja**: Simple, rápido
**Desventaja**: Pierde orden y estructura sintáctica

#### 2. TF-IDF Ponderado

$$
\vec{d} = \frac{1}{\sum_{t} \text{TF-IDF}(t)} \sum_{w \in d} \text{TF-IDF}(w) \cdot \vec{e}_w
$$

Pondera las palabras por su importancia en el documento.

#### 3. Modelos de Sentencia (Sentence-BERT, USE)

Modelos neuronales entrenados específicamente para producir embeddings a nivel de oración o párrafo:

$$
\vec{d} = \text{SBERT}(\text{texto})
$$

Producen vectores que capturan significado composicional de forma nativa.

### Normalización de Vectores

Es común normalizar los vectores a longitud unitaria:

$$
\hat{v} = \frac{v}{\|v\|}
$$

Esto permite que el producto punto sea equivalente al coseno:

$$
\hat{u} \cdot \hat{v} = \cos(\theta)
$$

---

## Métricas de Similitud y Distancia

### 1. Similitud Coseno

Mide el coseno del ángulo entre dos vectores, ignorando magnitudes:

$$
\text{similitud}_{\cos}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}}
$$

**Propiedades:**
- Rango: $[-1, 1]$
- 1: vectores en la misma dirección
- 0: vectores ortogonales (no relacionados)
- -1: vectores en direcciones opuestas

**Cuándo usarla**: Cuando la dirección del vector es más importante que su magnitud (típico en NLP).

### 2. Distancia Euclidiana

Mide la distancia geométrica directa entre vectores:

$$
d(A, B) = \|A - B\| = \sqrt{\sum_{i=1}^{d} (A_i - B_i)^2}
$$

**Relación con similitud coseno**: Si los vectores están normalizados a norma 1:

$$
d(A, B)^2 = 2 - 2 \cdot \text{similitud}_{\cos}(A, B)
$$

**Cuándo usarla**: Cuando la magnitud del vector tiene significado semántico.

### 3. Producto Punto (Dot Product)

$$
A \cdot B = \sum_{i=1}^{d} A_i B_i = \|A\| \|B\| \cos(\theta)
$$

Si los vectores están normalizados: $A \cdot B = \cos(\theta)$

**Ventaja**: Más rápido de calcular que coseno (no requiere división)

### 4. Distancia de Manhattan (L1)

$$
d_1(A, B) = \sum_{i=1}^{d} |A_i - B_i|
$$

Menos sensible a outliers que la euclidiana.

### Tabla Comparativa

| Métrica | Fórmula | Normalizada | Sensibilidad |
|---------|---------|-------------|--------------|
| Coseno | $\frac{A \cdot B}{\|A\| \|B\|}$ | Sí | Dirección |
| Euclidiana | $\|A - B\|$ | No | Magnitud + dirección |
| Producto punto | $A \cdot B$ | No | Magnitud × dirección |
| Manhattan | $\sum \|A_i - B_i\|$ | No | Magnitud + dirección |

---

## Indexación Vectorial

### El Problema de la Fuerza Bruta

Para $N$ documentos de dimensión $d$, buscar los $k$ vecinos más cercanos por fuerza bruta:

$$
O(N \cdot d)
$$

Con $N = 1,000,000$ y $d = 768$, esto es inviable en tiempo real.

### Estrategias de Aproximación (ANN)

**ANN** (Approximate Nearest Neighbors) sacrifica exactitud por velocidad:

#### 1. Locality Sensitive Hashing (LSH)

Usa funciones hash que colisionan con alta probabilidad para vectores similares:

$$
h(v) = \text{sign}(v \cdot r)
$$

Donde $r$ es un vector aleatorio. Vectores similares tienen mayor probabilidad de compartir el mismo hash.

#### 2. HNSW (Hierarchical Navigable Small World)

Construye un grafo de proximidad en múltiples capas:

- **Capa 0**: todos los vectores con conexiones locales
- **Capas superiores**: subsets espaciados con conexiones de largo alcance

Complejidad de búsqueda: $O(\log N)$ aproximadamente.

#### 3. Índice Invertido con Product Quantization (IVFPQ)

**Paso 1 - Clustering (Voronoi)**:

Divide el espacio en $k$ regiones usando k-means:

$$
C = \{c_1, c_2, \dots, c_k\}
$$

Cada vector se asigna al centroide más cercano.

**Paso 2 - Quantization**:

Comprime los vectores residuales usando codebooks:

$$
v \approx c_i + q_1 + q_2 + \dots + q_m
$$

Donde $q_j$ son vectores de un codebook pequeño.

**Búsqueda**:
1. Encontrar los $nprobe$ centroides más cercanos a la query
2. Buscar solo dentro de esos clusters
3. Re-rankear los candidatos exactamente

---

## Recuperación y Ranking

### Búsqueda por Similitud

Dada una query vectorial $q$ y un índice de documentos $D = \{d_1, \dots, d_N\}$:

$$
\text{rank}(d_i) = \text{similitud}(q, d_i)
$$

Retornamos los $k$ documentos con mayor similitud:

$$
\text{top-}k = \arg\max_{d_i \in D}^{(k)} \text{similitud}(q, d_i)
$$

### Score de Relevancia Combinado

En sistemas híbridos, se puede combinar similitud semántica con matching léxico:

$$
\text{score}_{\text{final}}(q, d) = \alpha \cdot \text{sim}_{\text{semántica}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{léxico}}(q, d)
$$

Donde $\alpha \in [0, 1]$ pondera la contribución semántica.

### Filtrado Post-Recuperación

A veces se aplican filtros después de la recuperación vectorial:
- Metadatos (fecha, autor, categoría)
- Deduplicación
- Re-ranking por modelo más pesado (cross-encoder)

---

## Re-ranking y Refinamiento

### Bi-encoders vs. Cross-encoders

| Tipo | Arquitectura | Velocidad | Precisión |
|------|-------------|-----------|-----------|
| **Bi-encoder** | Codifica query y doc por separado, luego compara | Muy rápido | Menor |
| **Cross-encoder** | Codifica query y doc juntos en un solo modelo | Lento | Mayor |

### Pipeline de Re-ranking

```
1. Bi-encoder recupera top-100 candidatos rápidamente
2. Cross-encoder re-calcula score para los 100 candidatos
3. Re-ordena por el nuevo score más preciso
4. Retorna top-10 final
```

### Fórmula de Re-ranking

$$
\text{score}_{\text{final}}(q, d) = \text{CrossEncoder}(q, d)
$$

Donde el cross-encoder recibe la concatenación `[CLS] query [SEP] document [SEP]` y produce un score de relevancia.

---

## Requerimientos y Consideraciones

### Requerimientos de Datos

| Componente | Requerimiento |
|------------|---------------|
| Embeddings | Corpus de entrenamiento o modelo pre-entrenado |
| Indexación | Memoria proporcional a $N \times d$ |
| Búsqueda | Latencia objetivo típicamente < 100ms |

### Trade-offs de Indexación ANN

| Parámetro | Alto | Bajo |
|-----------|------|------|
| `nprobe` (IVF) | Más exacto, más lento | Menos exacto, más rápido |
| `nlist` (IVF) | Menos vectores por cluster | Más vectores por cluster |
| `ef_search` (HNSW) | Más exacto, más lento | Menos exacto, más rápido |
| `M` (HNSW) | Más conexiones, más memoria | Menos conexiones, menos memoria |

### Complejidad Computacional

| Operación | Complejidad |
|-----------|-------------|
| Embedding de query | $O(L \cdot d^2)$ donde $L$ = longitud |
| Indexación secuencial | $O(N \cdot d)$ |
| Búsqueda fuerza bruta | $O(N \cdot d)$ |
| Búsqueda ANN (HNSW) | $O(\log N \cdot d)$ |
| Búsqueda IVF | $O(\frac{N}{k} \cdot d)$ donde $k$ = clusters |

---

## Resumen para Examen

### Conceptos Clave

| Concepto | Definición |
|----------|------------|
| **Búsqueda semántica** | Recuperación basada en significado vectorial, no palabras exactas |
| **Embedding de documento** | Vector denso que representa el significado de un texto |
| **ANN** | Algoritmos de vecinos más cercanos aproximados para escalabilidad |
| **HNSW** | Grafo jerárquico de proximidad para búsqueda eficiente |
| **IVF** | Indexación por clusters (Voronoi) con búsqueda local |
| **Bi-encoder** | Codifica query y doc por separado |
| **Cross-encoder** | Codifica query y doc juntos, más preciso pero lento |

### Fórmulas Importantes

$$
\text{similitud}_{\cos}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

$$
\vec{d}_{\text{promedio}} = \frac{1}{|d|} \sum_{w \in d} \vec{e}_w
$$

$$
\text{score}_{\text{híbrido}} = \alpha \cdot \text{sim}_{\text{sem}} + (1 - \alpha) \cdot \text{score}_{\text{léc}}
$$

### Preguntas de Examen Frecuentes

**1. ¿Cuál es la diferencia fundamental entre búsqueda léxica y semántica?**
La búsqueda léxica coincide términos exactos; la semántica busca similitud de significado en espacio vectorial.

**2. ¿Por qué normalizamos vectores antes de calcular similitud?**
Para que la comparación dependa solo de la dirección (semántica), no de la magnitud (frecuencia).

**3. ¿Qué problema resuelven los índices ANN?**
Evitan el costo lineal $O(N)$ de la búsqueda por fuerza bruta en corpus grandes.

**4. ¿Cuándo preferir un bi-encoder sobre un cross-encoder?**
Cuando la velocidad es crítica y se puede tolerar menor precisión (fase de recuperación inicial).

**5. ¿Por qué el producto punto entre vectores normalizados equivale al coseno?**
Porque $A \cdot B = \|A\| \|B\| \cos(\theta)$, y si $\|A\| = \|B\| = 1$, entonces $A \cdot B = \cos(\theta)$.

**6. ¿Qué es el re-ranking y por qué es útil?**
Es aplicar un modelo más preciso pero lento (cross-encoder) solo a un subconjunto pequeño de candidatos recuperados por un modelo rápido (bi-encoder).

### Ventajas y Limitaciones

| Ventajas | Limitaciones |
|----------|--------------|
| Captura sinónimos y paráfrasis | Requiere modelos pre-entrenados |
| Maneja polisemia (con modelos contextuales) | Computacionalmente costoso |
| Permite consultas en lenguaje natural | Calidad depende del embedding |
| Escalable con ANN | Puede perderse en el espacio vectorial |
| Combinable con búsqueda léxica | "Caja negra": difícil explicar por qué un resultado |

---

## Ejercicios Propuestos

### Ejercicio 1: Similitud Coseno
Dados embeddings de dimensión 3:
- Query: $q = [1, 1, 0]$
- Doc A: $d_A = [1, 0, 1]$
- Doc B: $d_B = [2, 2, 0]$

Calcula similitud coseno de $q$ con ambos documentos. ¿Cuál es más similar?

**Solución:**
- $\|q\| = \sqrt{1 + 1 + 0} = \sqrt{2} \approx 1.414$
- $\|d_A\| = \sqrt{1 + 0 + 1} = \sqrt{2} \approx 1.414$
- $\|d_B\| = \sqrt{4 + 4 + 0} = \sqrt{8} \approx 2.828$

- $q \cdot d_A = 1 + 0 + 0 = 1$
- $\text{sim}(q, d_A) = 1 / (1.414 \cdot 1.414) = 1/2 = 0.5$

- $q \cdot d_B = 2 + 2 + 0 = 4$
- $\text{sim}(q, d_B) = 4 / (1.414 \cdot 2.828) = 4/4 = 1.0$

**Respuesta**: Doc B es más similar (similitud = 1.0). Notar que $d_B = 2q$, tienen la misma dirección.

### Ejercicio 2: Embedding de Documento por Promedio
Documento: `["el", "gato", "come"]`
Embeddings:
- "el" = [1, 0]
- "gato" = [0, 1]
- "come" = [1, 1]

Calcula el vector del documento.

**Solución:**

$$
\vec{d} = \frac{1}{3}([1,0] + [0,1] + [1,1]) = \frac{1}{3}[2, 2] = [0.667, 0.667]
$$

### Ejercicio 3: Distancia Euclidiana vs. Coseno
Vectores normalizados: $A = [0.6, 0.8]$, $B = [0.8, 0.6]$

Calcula similitud coseno y distancia euclidiana al cuadrado.

**Solución:**
- $\|A\| = \sqrt{0.36 + 0.64} = 1$ (ya normalizado)
- $\|B\| = \sqrt{0.64 + 0.36} = 1$ (ya normalizado)
- $A \cdot B = 0.48 + 0.48 = 0.96$
- $\text{similitud}_{\cos} = 0.96 / (1 \cdot 1) = 0.96$

- $A - B = [-0.2, 0.2]$
- $d(A,B)^2 = 0.04 + 0.04 = 0.08$
- Verificación: $d^2 = 2 - 2(0.96) = 2 - 1.92 = 0.08$ ✓

### Ejercicio 4: Análisis Crítico
¿Por qué la búsqueda semántica podría fallar al buscar "código de vestimenta" en un corpus de programación?

**Respuesta**: Porque "código" en el corpus probablemente está asociado a "programación", "fuente", "compilación", etc. El embedding promedio del documento "código de vestimenta" podría ser dominado por el sentido de "código" como software, llevando a falsos positivos.

---

## Referencias para Estudio

- Manning, C. D. et al. *Introduction to Information Retrieval*, Capítulos 6-7
- Johnson, J. et al. "Billion-scale similarity search with GPUs" (FAISS, 2017)
- Malkov, Y. & Yashunin, D. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
- Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
- Karpukhin, V. et al. "Dense Passage Retrieval for Open-Domain Question Answering" (2020)
