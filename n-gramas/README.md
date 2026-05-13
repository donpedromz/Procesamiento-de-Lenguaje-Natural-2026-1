# Modelos de Lenguaje: N-gramas y Bigramas

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Fundamentos de Probabilidad](#fundamentos-de-probabilidad)
3. [Aproximación Markoviana](#aproximación-markoviana)
4. [Modelo de Bigramas](#modelo-de-bigramas)
5. [Estimación MLE (Maximum Likelihood)](#estimación-mle)
6. [Suavizado de Laplace](#suavizado-de-laplace)
7. [Log-Probabilidad y Underflow](#log-probabilidad-y-underflow)
8. [Generación de Texto](#generación-de-texto)
9. [Perplejidad](#perplejidad)
10. [Resumen para Examen](#resumen-para-examen)
11. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

Un **modelo de lenguaje** (Language Model, LM) asigna una probabilidad a una secuencia de palabras. Formalmente, estima $P(w_1, w_2, \dots, w_n)$.

### Aplicaciones
- Reconocimiento de voz (predicción de la siguiente palabra)
- Traducción automática
- Corrección ortográfica
- Generación de texto
- Autocompletado

> **Idea central**: Si podemos estimar qué tan probable es una secuencia de palabras, podemos comparar alternativas y elegir la más "natural".

---

## Fundamentos de Probabilidad

### Regla de la Cadena (Chain Rule)

La probabilidad conjunta se descompone como:

$$
P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, \dots, w_{i-1})
$$

**Ejemplo para 3 palabras:**

$$
P(w_1, w_2, w_3) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2)
$$

### El Problema

Para la palabra $w_{100}$, necesitaríamos $P(w_{100} | w_1, \dots, w_{99})$. El número de posibles historias es $|V|^{99}$, imposible de estimar con datos finitos.

---

## Aproximación Markoviana

### Hipótesis de Markov

Asumimos que el contexto relevante está limitado a las $N-1$ palabras anteriores:

$$
P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-N+1}, \dots, w_{i-1})
$$

Esto define los **modelos de N-gramas**:

| Orden | Nombre | Contexto | Fórmula |
|-------|--------|----------|---------|
| 1 | Unigrama | Ninguno | $P(w_i)$ |
| 2 | Bigrama | 1 palabra | $P(w_i \| w_{i-1})$ |
| 3 | Trigrama | 2 palabras | $P(w_i \| w_{i-2}, w_{i-1})$ |
| N | N-grama | N-1 palabras | $P(w_i \| w_{i-N+1}, \dots)$ |

> **Pregunta de examen**: ¿Cuál es la hipótesis del modelo de bigramas?
> **Respuesta**: Cada palabra depende únicamente de la palabra inmediatamente anterior.

### Tokens Especiales

- **<s>**: inicio de oración (start)
- **</s>**: fin de oración (end)

Permiten modelar probabilidades para la primera y última palabra.

---

## Modelo de Bigramas

### Definición

Aproxima la probabilidad de una secuencia como:

$$
P(w_1, w_2, \dots, w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-1})
$$

Con $w_0 = \text{<s>}$.

### Ejemplo

Para la oración "yo como arepa":

$$
P(\text{"yo como arepa"}) \approx P(\text{"yo"} | \text{<s>}) \cdot P(\text{"como"} | \text{"yo"}) \cdot P(\text{"arepa"} | \text{"como"}) \cdot P(\text{</s>} | \text{"arepa"})
$$

---

## Estimación MLE

### Maximum Likelihood Estimation

Las probabilidades se estiman por frecuencias del corpus:

$$
P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

Donde:
- $C(w_{i-1}, w_i)$: frecuencia del bigrama
- $C(w_{i-1})$: frecuencia del unigrama como primera palabra

### Ejemplo de Corpus

```
[<s> yo como arepa </s>]
[<s> yo como arroz </s>]
[<s> yo quiero arepa </s>]
```

**Conteos:**

| Bigrama | Frecuencia |
|---------|-----------|
| (<s>, yo) | 3 |
| (yo, como) | 2 |
| (yo, quiero) | 1 |
| (como, arepa) | 1 |
| (como, arroz) | 1 |

**Cálculo:**

$$
P(\text{"arepa"} | \text{"como"}) = \frac{C(\text{"como"}, \text{"arepa"})}{C(\text{"como"})} = \frac{1}{2} = 0.5
$$

$$
P(\text{"arroz"} | \text{"como"}) = \frac{1}{2} = 0.5
$$

---

## Suavizado de Laplace

### El Problema del Cero

Si un bigrama nunca apareció en entrenamiento:

$$
C(w_{i-1}, w_i) = 0 \implies P(w_i | w_{i-1}) = 0
$$

Esto hace que toda la probabilidad de la secuencia sea cero.

### Fórmula de Laplace (Add-1 Smoothing)

$$
P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

Donde $|V|$ es el tamaño del vocabulario.

### Ejemplo con Laplace

Vocabulario: $\{\text{yo}, \text{como}, \text{quiero}, \text{arepa}, \text{arroz}, \text{<s>}, \text{</s>}\}$ → $|V| = 7$

Para $P(\text{"arepa"} | \text{"como"})$:

Sin Laplace:
$$
\frac{1}{2} = 0.5
$$

Con Laplace:
$$
\frac{1 + 1}{2 + 7} = \frac{2}{9} \approx 0.222
$$

Para un bigrama NO observado, como $P(\text{"perro"} | \text{"como"})$:

$$
\frac{0 + 1}{2 + 7} = \frac{1}{9} \approx 0.111
$$

### Interpretación
- Numerador: reserva una "cuenta" para cada bigrama posible
- Denominador: suma $|V|$ porque añadimos 1 por cada palabra del vocabulario
- Garantiza que $\sum_{w} P(w | w_{i-1}) = 1$

---

## Log-Probabilidad y Underflow

### Problema de Underflow

Multiplicar muchas probabilidades pequeñas ($< 0.1$) produce números tan pequeños que se redondean a 0 en punto flotante.

**Ejemplo:**

$$
0.1^{20} = 10^{-20} \approx 0 \text{ (en precisión simple)}
$$

### Solución: Usar Logaritmos

$$
\log P(w_1, \dots, w_n) = \sum_{i=1}^{n} \log P(w_i | w_{i-1})
$$

**Ventajas:**
1. Convierte producto en suma
2. Evita underflow
3. Mantiene el orden (log es monótono creciente)

---

## Generación de Texto

### Algoritmo

```
1. Iniciar con <s>
2. Mientras última palabra ≠ </s> y longitud < máximo:
   a. Obtener candidatos (palabras que siguen al contexto actual)
   b. Elegir aleatoriamente proporcional a frecuencias
   c. Añadir a la secuencia
3. Retornar secuencia sin <s> y </s>
```

### Muestreo

La probabilidad de elegir $w$ dado $w_{prev}$:

$$
P(\text{elegir } w) = \frac{C(w_{prev}, w)}{\sum_{w'} C(w_{prev}, w')}
$$

---

## Perplejidad

### Definición

La **perplejidad** (PP) es la métrica estándar para evaluar modelos de lenguaje:

$$
\text{PP}(W) = P(w_1, w_2, \dots, w_N)^{-1/N}
$$

En forma logarítmica:

$$
\text{PP}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{i-1})\right)
$$

Donde $N$ es el número total de tokens en el corpus de test.

### Interpretación

- Representa el número promedio de palabras entre las que el modelo "duda" al predecir la siguiente
- **Menor perplejidad = mejor modelo**
- Perplejidad = $|V|$: modelo uniforme (peor caso razonable)
- Perplejidad = 1: modelo perfecto (imposible en la práctica)

### Pregunta de Examen

**¿Por qué no podemos comparar perplejidades de modelos con vocabularios diferentes?**

Porque un vocabulario más grande aumenta la entropía base. Un modelo con $|V| = 10000$ naturalmente tendrá perplejidad más alta que uno con $|V| = 100$, incluso si ambos predicen igual de bien relativamente.

---

## Resumen para Examen

### Conceptos Clave a Memorizar

| Concepto | Definición |
|----------|------------|
| N-grama | Secuencia de N palabras consecutivas |
| Hipótesis de Markov | Una palabra solo depende de las N-1 anteriores |
| MLE | Estimar probabilidades por frecuencias observadas |
| Laplace smoothing | Añadir 1 a conteos para evitar probabilidad cero |
| Underflow | Pérdida de precisión al multiplicar muchos números pequeños |
| Log-prob | Usar logaritmos para convertir productos en sumas |
| Perplejidad | $\exp(-\frac{1}{N} \sum \log P)$, métrica de evaluación |

### Fórmulas Importantes

$$
P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

$$
P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

$$
\text{PP} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{i-1})\right)
$$

### Limitaciones de los N-gramas

1. **Contexto fijo**: No capturan dependencias a larga distancia
2. **Data sparsity**: Requieren corpus masivos
3. **Sin generalización**: "comer" y "comiendo" son tokens completamente diferentes
4. **Storage**: $O(|V|^N)$ parámetros para N-gramas

### Comparación de Modelos

| Modelo | Ventajas | Desventajas |
|--------|----------|-------------|
| Unigrama | Simple, pocos parámetros | Sin contexto |
| Bigrama | Captura dependencias locales | Contexto limitado |
| Trigrama | Mejor contexto | Más datos necesarios |
| N-grama (N>3) | Contexto más rico | Sparse data, almacenamiento masivo |

---

## Ejercicios Propuestos

### Ejercicio 1: Conteos
Dado el corpus:
```
[<s> el gato come pescado </s>]
[<s> el perro come carne </s>]
[<s> el gato duerme </s>]
```

Calcula:
1. $C(\text{"el"})$
2. $C(\text{"el"}, \text{"gato"})$
3. $C(\text{"come"})$

**Respuestas:**
1. 3 (aparece 3 veces como inicio)
2. 2
3. 2

### Ejercicio 2: Probabilidades MLE
Usando el corpus anterior:

1. $P(\text{"gato"} | \text{"el"})$
2. $P(\text{"pescado"} | \text{"come"})$
3. $P(\text{"carne"} | \text{"come"})$

**Respuestas:**
1. 2/3
2. 1/2
3. 1/2

### Ejercicio 3: Laplace Smoothing
Vocabulario: $\{\text{el}, \text{gato}, \text{perro}, \text{come}, \text{pescado}, \text{carne}, \text{duerme}, \text{<s>}, \text{</s>}\}$ → $|V| = 9$

Calcula $P_{\text{Laplace}}(\text{"pez"} | \text{"come"})$ donde "pez" no está en el corpus.

**Respuesta:**
$$
\frac{0 + 1}{2 + 9} = \frac{1}{11} \approx 0.091
$$

### Ejercicio 4: Log-Probabilidad
Calcula la log-probabilidad de "el gato come" usando MLE.

**Solución:**

$$
\log P = \log P(\text{"el"}|\text{<s>}) + \log P(\text{"gato"}|\text{"el"}) + \log P(\text{"come"}|\text{"gato"})
$$

$$
= \log(1) + \log(2/3) + \log(1/2) = 0 + (-0.405) + (-0.693) = -1.098
$$

---

## Referencias para Estudio

- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Capítulo 3
- Chen, S. F. & Goodman, J. "An empirical study of smoothing techniques for language modeling" (1999)
- Manning, C. D. & Schütze, H. *Foundations of Statistical Natural Language Processing*, Capítulo 6
