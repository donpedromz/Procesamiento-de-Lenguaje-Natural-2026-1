# Modelos de Lenguaje N-gramas

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Modelo de Bigramas](#modelo-de-bigramas)
4. [Suavizado de Laplace](#suavizado-de-laplace)
5. [Generación de Texto](#generación-de-texto)
6. [Evaluación de Modelos de Lenguaje](#evaluación-de-modelos-de-lenguaje)
7. [Requerimientos y Consideraciones](#requerimientos-y-consideraciones)
8. [Ejemplo Paso a Paso](#ejemplo-paso-a-paso)

---

## Introducción

Los **modelos de lenguaje** (Language Models, LM) son modelos probabilísticos que asignan una probabilidad a una secuencia de palabras. Son fundamentales en múltiples tareas de PLN:

- Reconocimiento de voz
- Traducción automática
- Corrección ortográfica
- Generación de texto
- Autocompletado

> **Definición formal**: Un modelo de lenguaje estima $P(w_1, w_2, \dots, w_n)$, la probabilidad conjunta de una secuencia de $n$ palabras.

---

## Fundamentos Teóricos

### Regla de la Cadena (Chain Rule)

La probabilidad conjunta de una secuencia se descompone usando la regla de la cadena de probabilidad condicional:

$$
P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) \cdots P(w_n | w_1, \dots, w_{n-1})
$$

Formalmente:

$$
P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, \dots, w_{i-1})
$$

### El Problema de la Longitud Arbitraria

El problema es que el contexto crece indefinidamente. Para la palabra $w_{100}$, necesitaríamos condicionar en las 99 palabras anteriores, lo que resulta en:

- **Explosión combinatoria**: El número de posibles historias es $|V|^{n-1}$
- **Data sparsity**: La mayoría de las secuencias largas nunca aparecen en el corpus de entrenamiento

### Aproximación Markoviana

La solución es asumir que el contexto relevante está limitado a las $N-1$ palabras anteriores (**hipótesis de Markov**):

$$
P(w_i | w_1, w_2, \dots, w_{i-1}) \approx P(w_i | w_{i-N+1}, \dots, w_{i-1})
$$

Esto define los **modelos de N-gramas**:

| Modelo | Contexto | Fórmula |
|--------|----------|---------|
| Unigrama | ninguno | $P(w_i)$ |
| Bigrama | 1 palabra | $P(w_i \| w_{i-1})$ |
| Trigrama | 2 palabras | $P(w_i \| w_{i-2}, w_{i-1})$ |
| N-grama | $N-1$ palabras | $P(w_i \| w_{i-N+1}, \dots, w_{i-1})$ |

---

## Modelo de Bigramas

### Definición
Un **modelo de bigramas** aproxima la probabilidad de una secuencia asumiendo que cada palabra depende únicamente de la palabra inmediatamente anterior:

$$
P(w_1, w_2, \dots, w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-1})
$$

Con la convención de que $w_0 = \text{<s>}$ (token de inicio de oración).

### Estimación de Probabilidades

Las probabilidades condicionales se estiman por **máxima verosimilitud** (Maximum Likelihood Estimation, MLE) a partir de frecuencias en el corpus:

$$
P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

Donde:
- $C(w_{i-1}, w_i)$: número de veces que el bigrama $(w_{i-1}, w_i)$ aparece en el corpus
- $C(w_{i-1})$: número de veces que la palabra $w_{i-1}$ aparece como primera palabra de un bigrama (es decir, su frecuencia como unigrama)

### Tokens Especiales

Se añaden delimitadores de oración para modelar adecuadamente los límites:

- **<s>**: inicio de oración (start)
- **</s>**: fin de oración (end)

Esto permite estimar probabilidades para la primera y última palabra de cada oración.

### Estructuras de Datos

El modelo utiliza dos contadores:

1. **Bigramas**: `dict[w_prev][w_curr] = count`
   - Mapea cada palabra a un contador de sus palabras siguientes

2. **Unigramas**: `dict[w] = count`
   - Cuenta las ocurrencias de cada palabra (como antecedente)

### Log-Probabilidad

Para evitar **underflow numérico** (producto de muchas probabilidades pequeñas → 0 en punto flotante), se trabaja con logaritmos:

$$
\log P(w_1, \dots, w_n) = \sum_{i=1}^{n} \log P(w_i | w_{i-1})
$$

Esto convierte el producto en suma, estabilizando numéricamente los cálculos.

---

## Suavizado de Laplace

### El Problema del Cero

Si un bigrama $(w_{i-1}, w_i)$ nunca apareció en el corpus de entrenamiento:

$$
C(w_{i-1}, w_i) = 0 \implies P(w_i | w_{i-1}) = 0
$$

Esto hace que toda la probabilidad de la secuencia sea cero, independientemente de las demás palabras.

### Suavizado de Laplace (Add-1 Smoothing)

Consiste en sumar 1 a todas las frecuencias de bigramas y ajustar el denominador para que las probabilidades sigan sumando 1:

$$
P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

Donde:
- $|V|$: tamaño del vocabulario (número de palabras únicas)
- Se suma $|V|$ al denominador porque se añade 1 a $|V|$ posibles bigramas $(w_{i-1}, w)$ para cada $w \in V$

### Interpretación del Suavizado

- **Numerador**: $C(w_{i-1}, w_i) + 1$ — reserva una "cuenta" para bigramas no observados
- **Denominador**: $C(w_{i-1}) + |V|$ — normaliza para mantener $\sum_{w} P(w | w_{i-1}) = 1$

### Limitaciones del Laplace

1. **Distorsión excesiva**: Para bigramas raros pero observados, el suavizado puede ser demasiado agresivo
2. **No diferencia**: Trata igual a bigramas no observados y a bigramas raros
3. **Alternativas**: Para producción se prefieren:
   - **Add-k smoothing**: sumar $k < 1$
   - **Backoff (Katz backoff)**: usar trigramas si hay datos, si no bigramas, si no unigramas
   - **Kneser-Ney smoothing**: considera la diversidad de contextos
   - **Interpolación lineal**: combinación ponderada de n-gramas de distinto orden

---

## Generación de Texto

### Algoritmo de Generación

Una vez entrenado el modelo, se puede generar texto muestreando palabra por palabra:

```
1. Iniciar con <s>
2. Mientras la última palabra no sea </s> y no se exceda max_len:
   a. Obtener las palabras candidatas y sus frecuencias
   b. Elegir una palabra aleatoriamente proporcional a su frecuencia
   c. Añadir la palabra a la secuencia
3. Retornar la secuencia (sin <s> y </s>)
```

### Muestreo por Frecuencia

La probabilidad de elegir una palabra $w$ dado el contexto $w_{prev}$ es:

$$
P(\text{elegir } w) = \frac{C(w_{prev}, w)}{\sum_{w'} C(w_{prev}, w')} = \frac{C(w_{prev}, w)}{C(w_{prev})}
$$

Esto corresponde al muestreo de la distribución de bigramas sin suavizado.

---

## Evaluación de Modelos de Lenguaje

### Perplejidad (Perplexity)

La métrica estándar para evaluar modelos de lenguaje es la **perplejidad**, definida como:

$$
\text{PP}(W) = P(w_1, w_2, \dots, w_N)^{-1/N} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{i-1})\right)
$$

Donde $N$ es el número total de palabras (tokens) en el corpus de test.

**Interpretación**: La perplejidad representa el número promedio de palabras entre las que el modelo "duda" al predecir la siguiente. Menor perplejidad = mejor modelo.

### Propiedades de la Perplejidad

- **Perplejidad = $|V|$**: Modelo que predice uniformemente entre todas las palabras del vocabulario
- **Perplejidad = 1**: Modelo perfecto (nunca se equivoca)
- **Comparación**: Solo es válida entre modelos con el mismo vocabulario y corpus de test

---

## Requerimientos y Consideraciones

### Requerimientos de Entrada

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `oraciones` | `List[List[str]]` | Lista de oraciones tokenizadas |
| `ventana` | — | En bigramas siempre es 1 (dependencia de 1 palabra previa) |

### Complejidad Computacional

| Operación | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Entrenamiento | $O(N)$ | $O(|V|^2)$ en peor caso |
| Probabilidad de bigrama | $O(1)$ | — |
| Log-prob de oración | $O(L)$ | — |
| Generación de texto | $O(L \cdot |V|)$ | — |

Donde:
- $N$: número total de tokens en el corpus de entrenamiento
- $|V|$: tamaño del vocabulario
- $L$: longitud de la oración a evaluar/generar

### Consideraciones Prácticas

1. **Tamaño del vocabulario**: Bigramas requieren $O(|V|^2)$ espacio. Para vocabularios grandes (>50k palabras), considerar técnicas de pruning (eliminar bigramas con frecuencia < umbral).

2. **Datos de entrenamiento**: Los modelos de N-gramas requieren cantidades masivas de texto para funcionar bien. Google N-gram usó billones de tokens.

3. **Contexto fijo**: Los bigramas no capturan dependencias a larga distancia. Para español, los trigramas o 4-gramas suelen funcionar mejor.

---

## Ejemplo Paso a Paso

### Corpus de Entrenamiento

```python
oraciones = [
    ["yo", "como", "arepa"],
    ["yo", "como", "arroz"],
    ["yo", "quiero", "arepa"],
    ["yo", "quiero", "hamburguesa"],
    ["yo", "quiero", "jager"],
    ["yo", "como", "perro"],
]
```

### Paso 1: Añadir delimitadores

```
[<s>, "yo", "como", "arepa", </s>]
[<s>, "yo", "como", "arroz", </s>]
...
```

### Paso 2: Contar bigramas

| Bigrama | Frecuencia |
|---------|-----------|
| (<s>, yo) | 6 |
| (yo, como) | 3 |
| (yo, quiero) | 3 |
| (como, arepa) | 1 |
| (como, arroz) | 1 |
| (como, perro) | 1 |
| ... | ... |

### Paso 3: Calcular probabilidad (sin suavizado)

$$
P(\text{"arepa"} | \text{"como"}) = \frac{C(\text{"como"}, \text{"arepa"})}{C(\text{"como"})} = \frac{1}{3} \approx 0.333
$$

### Paso 4: Calcular probabilidad (con Laplace)

Vocabulario: $\{\text{yo}, \text{como}, \text{quiero}, \text{arepa}, \text{arroz}, \text{hamburguesa}, \text{jager}, \text{perro}, \text{<s>}, \text{</s>}\}$ → $|V| = 10$

$$
P_{\text{Laplace}}(\text{"arepa"} | \text{"como"}) = \frac{1 + 1}{3 + 10} = \frac{2}{13} \approx 0.154
$$

### Paso 5: Calcular log-probabilidad de oración

Para `["yo", "como", "arepa"]`:

$$
\log P = \log P(\text{"yo"} | \text{<s>}) + \log P(\text{"como"} | \text{"yo"}) + \log P(\text{"arepa"} | \text{"como"}) + \log P(\text{</s>} | \text{"arepa"})
$$

---

## Referencias

- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Cap. 3
- Chen, S. F. & Goodman, J. "An empirical study of smoothing techniques for language modeling" (1999)
- Manning, C. D. & Schütze, H. *Foundations of Statistical Natural Language Processing*, Cap. 6
