# Word2Vec: Representaciones Vectoriales de Palabras

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Hipótesis Distribucional](#hipótesis-distribucional)
3. [One-Hot vs. Embeddings Densos](#one-hot-vs-embeddings-densos)
4. [Arquitectura Skip-gram](#arquitectura-skip-gram)
5. [Ventana de Contexto](#ventana-de-contexto)
6. [Forward Pass](#forward-pass)
7. [Función de Pérdida](#función-de-pérdida)
8. [Backpropagation](#backpropagation)
9. [Similitud Coseno](#similitud-coseno)
10. [Ventajas y Limitaciones](#ventajas-y-limitaciones)
11. [Resumen para Examen](#resumen-para-examen)
12. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

**Word2Vec** es una familia de modelos neuronales superficiales (dos capas) que aprenden representaciones vectoriales densas de palabras a partir de grandes corpus de texto.

> **Diferencia clave con TF-IDF**: Mientras TF-IDF produce vectores dispersos y de alta dimensionalidad (tamaño del vocabulario), Word2Vec produce vectores densos y de baja dimensionalidad (tipicamente 50-300 dimensiones).

### Motivación

- Las representaciones tradicionales (BoW, TF-IDF) no capturan significado semántico
- "rey" y "reina" deberían estar cerca en el espacio vectorial
- "Madrid" debería estar cerca de "España" y lejos de "programación"

---

## Hipótesis Distribucional

### Enunciado

> **"Conocerás cada palabra por las compañías que frecuenta"**
> — J. R. Firth (1957)

### Interpretación

El significado de una palabra puede inferirse por el contexto en el que aparece. Palabras que aparecen en contextos similares tienen significados similares.

**Ejemplo:**
- "El **gato** come pescado"
- "El **perro** come carne"

"gato" y "perro" aparecen en contextos similares (ambos son sujetos de "come"), por tanto deberían tener embeddings similares.

---

## One-Hot vs. Embeddings Densos

### Representación One-Hot

Cada palabra es un vector de dimensión $|V|$ con un 1 en su posición y 0 en el resto:

$$
\text{gato} = [0, 0, 1, 0, \dots, 0]
$$

**Problemas:**
- Dimensión muy alta ($|V|$ puede ser 100,000+)
- Todos los vectores son ortogonales (similitud coseno = 0 entre cualquier par)
- No captura relaciones semánticas

### Embeddings Densos (Word2Vec)

Cada palabra es un vector de dimensión $d \ll |V|$ (ej: $d = 100$):

$$
\text{gato} = [0.2, -0.5, 0.8, \dots, 0.1] \in \mathbb{R}^d
$$

**Ventajas:**
- Dimensión fija y manejable
- Palabras similares tienen vectores cercanos
- Capturan analogías semánticas y sintácticas

---

## Arquitectura Skip-gram

### Definición

El modelo **Skip-gram** predice las palabras del **contexto** dada una palabra **central** (target).

```
Input: palabra central → Output: palabras del contexto
```

> **Diferencia con CBOW**: CBOW predice la palabra central dado el contexto (lo inverso).

### Arquitectura de Red

```
Input (one-hot)     Hidden Layer        Output (softmax)
     │                   │                    │
     ▼                   ▼                    ▼
┌─────────┐        ┌─────────┐         ┌─────────┐
│  |V|    │        │   d     │         │   |V|   │
│  dim    │───────▶│  dim    │────────▶│  dim    │
│ (sparse)│   W    │ (dense) │   W2    │(softmax)│
└─────────┘        └─────────┘         └─────────┘
   w_t               embedding           P(w|w_t)
```

### Parámetros del Modelo

- **$W \in \mathbb{R}^{|V| \times d}$**: matriz de embeddings (input → hidden)
- **$W_2 \in \mathbb{R}^{d \times |V|}$**: matriz de contexto (hidden → output)
- **$d$**: dimensión de los embeddings (hiperparámetro)

> **La matriz $W$ es la que nos interesa**: cada fila $W[i]$ es el embedding de la palabra $i$.

---

## Ventana de Contexto

### Definición

La **ventana de contexto** ($k$) define cuántas palabras a cada lado de la palabra central se consideran contexto.

**Ejemplo con ventana = 2:**

```
oración: el perro come carne en el plato
              ↑
         palabra central: "come"

contexto: ["perro", "carne"]   (2 palabras a cada lado)
```

### Pares de Entrenamiento

Para cada posición $i$ en una oración, generamos pares $(w_i, w_j)$ para todo $j$ tal que $0 < |i - j| \leq k$.

---

## Forward Pass

### Paso 1: Obtener Embedding

Dado índice de palabra central $c$:

$$
h = W[c] \in \mathbb{R}^d
$$

Esto selecciona la fila $c$ de la matriz $W$ (lookup).

### Paso 2: Proyección al Vocabulario

$$
z = W_2^T \cdot h \in \mathbb{R}^{|V|}
$$

### Paso 3: Softmax

Convierte scores en probabilidades:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{|V|} e^{z_j}}
$$

**Propiedades del softmax:**
- $\sum_i y_i = 1$
- $0 < y_i < 1$
- Monótono: mayor $z_i$ → mayor $y_i$

### Estabilidad Numérica

Para evitar overflow en $e^{z_i}$ con $z_i$ grandes:

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

Restar el máximo no cambia el resultado pero estabiliza numéricamente.

---

## Función de Pérdida

### Entropía Cruzada (Cross-Entropy)

Para un par $(w_c, w_t)$ donde $w_c$ es central y $w_t$ es target de contexto:

La distribución real es one-hot (1 en la posición de $w_t$, 0 en el resto).

La pérdida para un solo ejemplo:

$$
\mathcal{L} = -\log P(w_t | w_c) = -\log \left( \frac{e^{z_t}}{\sum_j e^{z_j}} \right)
$$

Donde $z_t$ es el score para la palabra target.

### Pérdida Total

Para todos los pares de entrenamiento:

$$
\mathcal{L}_{\text{total}} = -\sum_{(c,t) \in \text{pares}} \log P(w_t | w_c)
$$

> **Interpretación**: Minimizar la pérdida equivale a maximizar la probabilidad de observar las palabras de contexto reales.

---

## Backpropagation

### Gradiente respecto a $W_2$

Definimos el error en la capa de salida:

$$
e = y - \hat{y}
$$

Donde $y$ es one-hot (1 en target, 0 en resto) y $\hat{y}$ es la predicción softmax.

El gradiente para $W_2$:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = h \cdot e^T
$$

Actualización:

$$
W_2 \leftarrow W_2 - \eta \cdot h \cdot e^T
$$

### Gradiente respecto a $W$

Propagamos el error hacia atrás:

$$
\delta_h = W_2 \cdot e
$$

Actualización para la fila $c$ de $W$:

$$
W[c] \leftarrow W[c] - \eta \cdot \delta_h
$$

> **Nota**: Solo se actualiza la fila correspondiente a la palabra central, no toda la matriz. Esto hace que el entrenamiento sea eficiente.

---

## Similitud Coseno

### Definición

Una vez entrenado el modelo, medimos la similitud entre palabras usando el **coseno del ángulo** entre sus vectores:

$$
\text{similitud}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}}
$$

### Propiedades

- Rango: $[-1, 1]$
- 1: vectores idénticos (misma dirección)
- 0: vectores ortogonales (no relacionados)
- -1: vectores opuestos

### Interpretación Geométrica

- Palabras similares → vectores cercanos → ángulo pequeño → coseno cercano a 1
- Palabras diferentes → vectores lejanos → ángulo grande → coseno cercano a 0

---

## Ventajas y Limitaciones

### Ventajas
1. **Captura semántica**: "rey" ≈ "reina" en el espacio
2. **Analogías**: $\vec{\text{rey}} - \vec{\text{hombre}} + \vec{\text{mujer}} \approx \vec{\text{reina}}$
3. **Dimensión fija**: independiente del tamaño del vocabulario
4. **Eficiente**: solo dos capas, entrenamiento rápido

### Limitaciones
1. **Palabras polisémicas**: Un solo vector para "banco" (financiero y de sentarse)
2. **Sin contexto dinámico**: El embedding es fijo independientemente del contexto
3. **Desconocidas en test**: Palabras no vistas en entrenamiento no tienen embedding
4. **Dependencia del corpus**: Sesgos del corpus se reflejan en los vectores

### Evolución Posterior

| Modelo | Mejora sobre Word2Vec |
|--------|----------------------|
| **GloVe** | Factorización de matriz de co-ocurrencia |
| **FastText** | Maneja palabras raras mediante n-gramas de caracteres |
| **ELMo** | Embeddings contextuales (dependen de la oración) |
| **BERT** | Representaciones profundas y bidireccionales |

---

## Resumen para Examen

### Conceptos Clave

| Concepto | Definición |
|----------|------------|
| **Embedding** | Representación vectorial densa de una palabra |
| **Skip-gram** | Predice contexto dada palabra central |
| **CBOW** | Predice palabra central dado contexto (inverso) |
| **Ventana** | Número de palabras a cada lado consideradas contexto |
| **Softmax** | Normaliza scores a probabilidades que suman 1 |
| **Cross-entropy** | Función de pérdida: $-\log P(\text{target} \| \text{central})$ |
| **Coseno** | $\frac{A \cdot B}{\|A\| \|B\|}$, medida de similitud |

### Fórmulas Importantes

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$$
\mathcal{L} = -\log P(w_t | w_c)
$$

$$
\text{similitud}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

### Preguntas de Examen Frecuentes

**1. ¿Cuál es la diferencia entre Skip-gram y CBOW?**
- Skip-gram: input = 1 palabra, output = múltiples palabras de contexto
- CBOW: input = múltiples palabras de contexto, output = 1 palabra central
- Skip-gram funciona mejor con corpus pequeños; CBOW es más rápido

**2. ¿Por qué la matriz $W$ nos da los embeddings?**
Porque $W[c]$ es el vector que representa a la palabra $c$ en la capa oculta, y este vector es compartido para predecir todos los contextos posibles.

**3. ¿Qué problema resuelve restar el máximo en softmax?**
Evita overflow numérico cuando $z_i$ es muy grande, ya que $e^{1000}$ excede el rango de punto flotante.

**4. ¿Por qué Word2Vec captura analogías?**
Porque las relaciones semánticas se codifican como direcciones en el espacio vectorial. La dirección "hombre → mujer" es similar a "rey → reina".

**5. ¿Qué limitación tiene Word2Vec respecto a BERT?**
Word2Vec asigna un único vector por palabra, sin importar el contexto. BERT genera representaciones contextuales que cambian según la oración.

---

## Ejercicios Propuestos

### Ejercicio 1: Cálculo de Softmax
Dado $z = [2.0, 1.0, 0.1]$ para un vocabulario de 3 palabras, calcula softmax.

**Solución:**

Paso 1: estabilizar restando máximo (2.0):
$z' = [0, -1.0, -1.9]$

Paso 2: exponenciar:
$e^{z'} = [1.0, 0.368, 0.150]$

Paso 3: suma = $1.0 + 0.368 + 0.150 = 1.518$

Paso 4: normalizar:
- $y_1 = 1.0 / 1.518 = 0.659$
- $y_2 = 0.368 / 1.518 = 0.242$
- $y_3 = 0.150 / 1.518 = 0.099$

Verificación: $0.659 + 0.242 + 0.099 = 1.0$ ✓

### Ejercicio 2: Similitud Coseno
Dados:
- $A = [1, 2, 3]$
- $B = [2, 4, 6]$

Calcula similitud coseno.

**Solución:**
- $A \cdot B = 1·2 + 2·4 + 3·6 = 2 + 8 + 18 = 28$
- $\|A\| = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$
- $\|B\| = \sqrt{4 + 16 + 36} = \sqrt{56} \approx 7.483$
- Similitud = $28 / (3.742 · 7.483) = 28 / 28.0 = 1.0$

**Interpretación**: $B = 2A$, son vectores paralelos (misma dirección).

### Ejercicio 3: Pares de Entrenamiento
Oración: `["el", "gato", "come", "pescado"]`
Ventana = 1

Genera todos los pares (central, contexto) para Skip-gram.

**Respuesta:**
- ("el", "gato")
- ("gato", "el")
- ("gato", "come")
- ("come", "gato")
- ("come", "pescado")
- ("pescado", "come")

Total: 6 pares.

**Pregunta**: ¿Cuántos pares con ventana = 2?

Para "gato": contexto = ["el", "come", "pescado"]
Para "come": contexto = ["el", "gato", "pescado"]

Total: 8 pares.

### Ejercicio 4: Análisis Crítico
¿Por qué Word2Vec podría dar embeddings similares para "banco" (sentarse) y "banco" (financiero)?

**Respuesta**: Porque usa un único vector por palabra. Si el corpus no tiene suficiente contexto diferenciador, o si ambos usos comparten contextos similares, el modelo promedia ambos significados en un solo vector.

---

## Referencias para Estudio

- Mikolov, T. et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
- Mikolov, T. et al. "Distributed Representations of Words and Phrases and their Compositionality" (2013)
- Goldberg, Y. & Levy, O. "word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method" (2014)
- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Capítulo 6
