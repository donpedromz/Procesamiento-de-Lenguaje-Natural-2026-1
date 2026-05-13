# Clasificación de Texto con Naive Bayes

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Teorema de Bayes](#teorema-de-bayes)
4. [Supuesto Naive (Ingenuo)](#supuesto-naive-ingenuo)
5. [Naive Bayes para Texto](#naive-bayes-para-texto)
6. [Suavizado de Laplace](#suavizado-de-laplace)
7. [Representación Bag of Words](#representación-bag-of-words)
8. [Pipeline Completo](#pipeline-completo)
9. [Requerimientos y Consideraciones](#requerimientos-y-consideraciones)
10. [Ejemplo de Análisis de Sentimiento](#ejemplo-de-análisis-de-sentimiento)

---

## Introducción

La **clasificación de texto** es la tarea de asignar una o más categorías predefinidas a un documento de texto. Es una de las aplicaciones más comunes del PLN:

- **Análisis de sentimiento**: positivo / negativo / neutral
- **Detección de spam**: spam / no spam
- **Clasificación de noticias**: deportes / política / tecnología / entretenimiento
- **Enrutamiento de correos**: asignar a departamentos

> **Naive Bayes** es uno de los clasificadores más simples y efectivos para clasificación de texto, especialmente cuando se trabaja con datasets de tamaño moderado.

---

## Fundamentos Teóricos

### Definición del Problema

Dado:
- Un documento $d$ representado como una bolsa de palabras (bag of words)
- Un conjunto fijo de clases $C = \{c_1, c_2, \dots, c_k\}$

Encontrar:

$$
\hat{c} = \arg\max_{c \in C} P(c | d)
$$

La clase más probable dado el documento.

---

## Teorema de Bayes

El teorema de Bayes relaciona probabilidades condicionales:

$$
P(c | d) = \frac{P(d | c) \cdot P(c)}{P(d)}

$$

Donde:
- $P(c | d)$: **posterior** — probabilidad de la clase dado el documento
- $P(d | c)$: **verosimilitud** (likelihood) — probabilidad del documento dado la clase
- $P(c)$: **prior** — probabilidad a priori de la clase
- $P(d)$: **evidencia** — probabilidad del documento (constante para todas las clases)

Para clasificación, solo necesitamos el numerador, ya que $P(d)$ es igual para todas las clases:

$$
\hat{c} = \arg\max_{c \in C} P(d | c) \cdot P(c)
$$

---

## Supuesto Naive (Ingenuo)

### La Suposición de Independencia Condicional

Naive Bayes asume que las características (palabras en nuestro caso) son **condicionalmente independientes** dada la clase:

$$
P(d | c) = P(w_1, w_2, \dots, w_n | c) = \prod_{i=1}^{n} P(w_i | c)
$$

Esto significa que la presencia de una palabra en un documento no afecta la probabilidad de otras palabras, **una vez que conocemos la clase**.

> **Advertencia**: Esta suposición es claramente falsa en lenguaje natural. Por ejemplo, "New" y "York" no son independientes. Sin embargo, Naive Bayes funciona sorprendentemente bien a pesar de esta "ingenuidad".

### Por qué Funciona a Pesar de la Suposición Falsa

1. **Clasificación, no estimación**: Solo necesitamos ordenar las clases por probabilidad, no estimar las probabilidades exactas.
2. **Robustez**: Los errores de independencia a veces se cancelan entre sí.
3. **Eficiencia**: Requiere muchos menos parámetros que modelos que capturan dependencias.

---

## Naive Bayes para Texto

### Modelo Multinomial

Para clasificación de texto, se usa la variante **Multinomial Naive Bayes**, que modela la frecuencia de palabras:

$$
\hat{c} = \arg\max_{c \in C} P(c) \prod_{i=1}^{n} P(w_i | c)
$$

Donde $w_i$ son las palabras del documento (puede haber repeticiones).

### Estimación de Probabilidades

#### Prior $P(c)$

$$
P(c) = \frac{N_c}{N}
$$

Donde:
- $N_c$: número de documentos de entrenamiento de clase $c$
- $N$: número total de documentos de entrenamiento

#### Likelihood $P(w | c)$

$$
P(w | c) = \frac{\text{conteo}(w, c)}{\sum_{w' \in V} \text{conteo}(w', c)} = \frac{\text{conteo}(w, c)}{|D_c|}
$$

Donde:
- $\text{conteo}(w, c)$: número de veces que la palabra $w$ aparece en todos los documentos de clase $c$
- $|D_c|$: número total de tokens en documentos de clase $c$

### Versión Logarítmica

Para evitar underflow numérico (producto de muchas probabilidades pequeñas), se usa logaritmos:

$$
\hat{c} = \arg\max_{c \in C} \left[ \log P(c) + \sum_{i=1}^{n} \log P(w_i | c) \right]
$$

Esto convierte el producto en suma y mantiene el argmax invariante porque $\log$ es monótonamente creciente.

---

## Suavizado de Laplace

### El Problema de Palabras Desconocidas

Si una palabra del documento de test nunca apareció en los documentos de entrenamiento de una clase:

$$
P(w | c) = 0 \implies \prod_{i} P(w_i | c) = 0
$$

Esto anula toda la evidencia de otras palabras.

### Solución: Suavizado Add-1

$$
P_{\text{Laplace}}(w | c) = \frac{\text{conteo}(w, c) + 1}{|D_c| + |V|}
$$

Donde:
- $|V|$: tamaño del vocabulario total (todas las palabras vistas en cualquier clase)
- Se suma 1 al numerador por cada palabra del vocabulario
- Se suma $|V|$ al denominador para normalizar

### Para Palabras Fuera del Vocabulario (OOV)

Si una palabra $w_{\text{new}}$ nunca fue vista en entrenamiento:

$$
P(w_{\text{new}} | c) = \frac{0 + 1}{|D_c| + |V|} = \frac{1}{|D_c| + |V|}
$$

Esto asigna una probabilidad pequeña pero no cero.

---

## Representación Bag of Words

### Definición

**Bag of Words (BoW)** representa un documento como un vector de frecuencias de palabras, ignorando el orden y la estructura gramatical.

Dado un vocabulario $V = \{w_1, w_2, \dots, w_{|V|}\}$, un documento $d$ se representa como:

$$
\vec{d} = (f_1, f_2, \dots, f_{|V|})
$$

Donde $f_i$ es la frecuencia de la palabra $w_i$ en el documento.

### Relación con Naive Bayes

En el modelo multinomial, Naive Bayes usa las frecuencias directamente:

$$
P(d | c) \propto \prod_{i=1}^{|V|} P(w_i | c)^{f_i}
$$

Donde $f_i$ es la frecuencia de $w_i$ en el documento $d$.

---

## Pipeline Completo

```
┌─────────────────┐
│  Corpus Bruto   │  List[str] con documentos de entrenamiento
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Preprocesamiento│  normalizar → tokenizar → stopwords → stemming
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Entrenamiento │
│  Naive Bayes    │
└─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌─────────┐
│ Priors│ │ Likelihoods│
│log P(c)│ │ log P(w|c) │
└───────┘ └─────────┘
         │
         ▼
┌─────────────────┐
│   Clasificación │  argmax_c [log P(c) + Σ log P(w|c)]
│   de Nuevo Doc  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│    Predicción   │  c ∈ C (clase predicha)
└─────────────────┘
```

---

## Requerimientos y Consideraciones

### Requerimientos de Entrada

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `documentos` | `List[str]` | Documentos de entrenamiento |
| `etiquetas` | `List[str]` | Clases correspondientes |
| `texto_test` | `str` | Documento a clasificar |

### Complejidad Computacional

| Operación | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Entrenamiento | $O(N \cdot \bar{L} + |V| \cdot |C|)$ | $O(|V| \cdot |C|)$ |
| Clasificación | $O(L \cdot |C|)$ | $O(|C|)$ |

Donde:
- $N$: número de documentos de entrenamiento
- $\bar{L}$: longitud promedio de documento
- $|V|$: tamaño del vocabulario
- $|C|$: número de clases
- $L$: longitud del documento a clasificar

### Decisiones de Diseño

1. **Independencia de características**: El modelo no captura relaciones entre palabras ("no" + "bueno" = negativo, pero se tratan como independientes).

2. **Frecuencia vs. presencia**: Esta implementación usa frecuencias (multinomial). Una alternativa es Bernoulli Naive Bayes, que solo considera presencia/ausencia.

3. **Balance de clases**: Si las clases están desbalanceadas, el prior $P(c)$ favorecerá la clase mayoritaria. Considerar técnicas de balanceo si es necesario.

---

## Ejemplo de Análisis de Sentimiento

### Dataset de Entrenamiento

16 reseñas de restaurantes balanceadas (8 positivas, 8 negativas).

### Clases
- **POS**: reseña positiva
- **NEG**: reseña negativa

### Cálculo de Priors

$$
P(\text{POS}) = \frac{8}{16} = 0.5 \implies \log P(\text{POS}) = \log(0.5) \approx -0.693
$$

$$
P(\text{NEG}) = \frac{8}{16} = 0.5 \implies \log P(\text{NEG}) = \log(0.5) \approx -0.693
$$

### Cálculo de Likelihoods (con Laplace)

Para la palabra "excelente" en clase POS:
- Conteo("excelente", POS) = 2
- Total tokens en POS = 150
- $|V|$ = 80

$$
P(\text{"excelente"} | \text{POS}) = \frac{2 + 1}{150 + 80} = \frac{3}{230} \approx 0.013
$$

$$
\log P(\text{"excelente"} | \text{POS}) \approx \log(0.013) \approx -4.34
$$

### Clasificación de un Nuevo Documento

Documento: `"excelente historia"`

Después de preprocesamiento: `["excel", "histori"]`

**Score para POS:**

$$
\text{score}_{\text{POS}} = \log P(\text{POS}) + \log P(\text{"excel"} | \text{POS}) + \log P(\text{"histori"} | \text{POS})
$$

**Score para NEG:**

$$
\text{score}_{\text{NEG}} = \log P(\text{NEG}) + \log P(\text{"excel"} | \text{NEG}) + \log P(\text{"histori"} | \text{NEG})
$$

Predicción:

$$
\hat{c} = \arg\max \{\text{score}_{\text{POS}}, \text{score}_{\text{NEG}}\}
$$

---

## Referencias

- McCallum, A. & Nigam, K. "A Comparison of Event Models for Naive Bayes Text Classification" (1998)
- Manning, C. D. et al. *Introduction to Information Retrieval*, Cap. 13
- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Cap. 4
