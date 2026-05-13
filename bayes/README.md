# Clasificación de Texto con Naive Bayes

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Definición del Problema](#definición-del-problema)
3. [Teorema de Bayes](#teorema-de-bayes)
4. [El Supuesto Naive (Ingenuo)](#el-supuesto-naive-ingenuo)
5. [Naive Bayes Multinomial](#naive-bayes-multinomial)
6. [Estimación de Parámetros](#estimación-de-parámetros)
7. [Suavizado de Laplace](#suavizado-de-laplace)
8. [Versión Logarítmica](#versión-logarítmica)
9. [Pipeline Completo](#pipeline-completo)
10. [Resumen para Examen](#resumen-para-examen)
11. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

La **clasificación de texto** consiste en asignar categorías predefinidas a documentos de texto. Es una de las aplicaciones más importantes del PLN.

### Aplicaciones Comunes
- **Análisis de sentimiento**: positivo / negativo / neutral
- **Detección de spam**: spam / no spam (ham)
- **Clasificación de noticias**: deportes / política / tecnología
- **Enrutamiento de correos**: asignar a departamentos

> **Naive Bayes** es uno de los clasificadores más simples y sorprendentemente efectivos para texto, especialmente con datasets pequeños a medianos.

---

## Definición del Problema

### Entrada
- Un documento $d$ (string de texto)
- Un conjunto de clases $C = \{c_1, c_2, \dots, c_k\}$

### Salida
La clase más probable:

$$
\hat{c} = \arg\max_{c \in C} P(c | d)
$$

---

## Teorema de Bayes

### Fórmula Fundamental

$$
P(c | d) = \frac{P(d | c) \cdot P(c)}{P(d)}
$$

| Término | Nombre | Significado |
|---------|--------|-------------|
| $P(c \| d)$ | Posterior | Probabilidad de la clase dado el documento |
| $P(d \| c)$ | Likelihood (Verosimilitud) | Probabilidad del documento dado la clase |
| $P(c)$ | Prior | Probabilidad a priori de la clase |
| $P(d)$ | Evidencia | Probabilidad del documento (constante) |

### Para Clasificación

Como $P(d)$ es el mismo para todas las clases:

$$
\hat{c} = \arg\max_{c \in C} P(d | c) \cdot P(c)
$$

> **Truco clave**: Solo necesitamos comparar numeradores; no hace falta calcular $P(d)$.

---

## El Supuesto Naive (Ingenuo)

### Independencia Condicional

Naive Bayes asume que las características (palabras) son **condicionalmente independientes** dada la clase:

$$
P(d | c) = P(w_1, w_2, \dots, w_n | c) = \prod_{i=1}^{n} P(w_i | c)
$$

**Esto significa**: una vez que sabemos la clase, saber que "New" aparece no nos dice nada sobre si aparece "York".

> **Es falso en la realidad**: Las palabras en lenguaje natural tienen dependencias fuertes. Sin embargo, Naive Bayes funciona muy bien a pesar de esta suposición "ingenua".

### ¿Por qué Funciona?

1. **Clasificación, no estimación**: Solo necesitamos el $\arg\max$, no las probabilidades exactas
2. **Cancelación de errores**: Los errores de independencia a veces se compensan
3. **Menos parámetros**: No necesitamos estimar covarianzas entre palabras
4. **Robustez**: Con pocos datos, modelos simples generalizan mejor

---

## Naive Bayes Multinomial

### Fórmula de Clasificación

$$
\hat{c} = \arg\max_{c \in C} P(c) \prod_{i=1}^{n} P(w_i | c)
$$

Donde $w_i$ son las palabras del documento (con repeticiones permitidas).

### Representación Bag of Words

El documento se representa como frecuencias de palabras, ignorando orden:

$$
\vec{d} = (f_1, f_2, \dots, f_{|V|})
$$

Con frecuencias $f_i$ para cada palabra del vocabulario.

---

## Estimación de Parámetros

### 1. Prior $P(c)$

$$
P(c) = \frac{N_c}{N}
$$

Donde:
- $N_c$: número de documentos de entrenamiento de clase $c$
- $N$: número total de documentos

**Ejemplo:**
- 70 reseñas positivas, 30 negativas
- $P(\text{POS}) = 70/100 = 0.7$
- $P(\text{NEG}) = 30/100 = 0.3$

### 2. Likelihood $P(w | c)$

$$
P(w | c) = \frac{\text{conteo}(w, c)}{\sum_{w' \in V} \text{conteo}(w', c)} = \frac{\text{conteo}(w, c)}{|D_c|}
$$

Donde:
- $\text{conteo}(w, c)$: veces que $w$ aparece en documentos de clase $c$
- $|D_c|$: total de tokens en documentos de clase $c$

**Ejemplo:**
- En reseñas positivas, "excelente" aparece 15 veces
- Total de tokens en positivas: 1500
- $P(\text{"excelente"} | \text{POS}) = 15/1500 = 0.01$

---

## Suavizado de Laplace

### Problema de Palabras Desconocidas

Si una palabra del test no apareció en entrenamiento para una clase:

$$
P(w | c) = 0 \implies \prod_{i} P(w_i | c) = 0
$$

¡Toda la evidencia se anula!

### Solución: Add-1 Smoothing

$$
P_{\text{Laplace}}(w | c) = \frac{\text{conteo}(w, c) + 1}{|D_c| + |V|}
$$

Donde $|V|$ es el tamaño del vocabulario total.

### Ejemplo

- $\text{conteo}(\text{"delicioso"}, \text{POS}) = 5$
- $|D_{\text{POS}}| = 1000$
- $|V| = 500$

Sin Laplace:
$$
P = 5/1000 = 0.005
$$

Con Laplace:
$$
P = \frac{5 + 1}{1000 + 500} = \frac{6}{1500} = 0.004
$$

Para palabra desconocida:
$$
P = \frac{0 + 1}{1000 + 500} = \frac{1}{1500} \approx 0.00067
$$

---

## Versión Logarítmica

### ¿Por qué Logaritmos?

Multiplicar muchas probabilidades pequeñas causa **underflow numérico**:

$$
0.01^{100} \approx 10^{-200} \rightarrow 0 \text{ (en punto flotante)}
$$

### Fórmula Logarítmica

$$
\hat{c} = \arg\max_{c \in C} \left[ \log P(c) + \sum_{i=1}^{n} \log P(w_i | c) \right]
$$

**Propiedades usadas:**
- $\log(ab) = \log(a) + \log(b)$ → convierte producto en suma
- $\log$ es monótonamente creciente → preserva el $\arg\max$

### Ejemplo de Cálculo

Documento: "excelente servicio"

**Clase POS:**
- $\log P(\text{POS}) = \log(0.5) = -0.693$
- $\log P(\text{"excelente"} | \text{POS}) = -4.2$
- $\log P(\text{"servicio"} | \text{POS}) = -3.5$
- **Score POS** = $-0.693 - 4.2 - 3.5 = -8.393$

**Clase NEG:**
- $\log P(\text{NEG}) = -0.693$
- $\log P(\text{"excelente"} | \text{NEG}) = -8.0$ (rara en negativas)
- $\log P(\text{"servicio"} | \text{NEG}) = -4.0$
- **Score NEG** = $-0.693 - 8.0 - 4.0 = -12.693$

**Predicción**: POS (score mayor: -8.393 > -12.693)

> **Nota importante**: En log-espacio, valores MENOS negativos son MAYORES probabilidades.

---

## Pipeline Completo

```
┌─────────────────────────────────────────────┐
│  1. PREPROCESAMIENTO                        │
│  normalizar → tokenizar → stopwords → stem  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  2. ENTRENAMIENTO                           │
│  Para cada clase c:                         │
│    • Calcular prior: log P(c)               │
│    • Contar tokens por clase                │
│    • Calcular likelihoods: log P(w|c)       │
│      (con Laplace smoothing)                │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  3. CLASIFICACIÓN                           │
│  Para documento nuevo d:                    │
│    • Preprocesar d                          │
│    • Para cada clase:                       │
│      score[c] = log P(c) + Σ log P(w|c)     │
│    • Retornar argmax score[c]               │
└─────────────────────────────────────────────┘
```

---

## Resumen para Examen

### Fórmulas Fundamentales

| Concepto | Fórmula |
|----------|---------|
| Bayes | $P(c \| d) = \frac{P(d \| c) P(c)}{P(d)}$ |
| Naive assumption | $P(d \| c) = \prod P(w_i \| c)$ |
| Clasificación | $\hat{c} = \arg\max_c P(c) \prod P(w_i \| c)$ |
| Prior | $P(c) = N_c / N$ |
| Likelihood | $P(w \| c) = \text{conteo}(w,c) / \|D_c\|$ |
| Laplace | $P(w \| c) = (\text{conteo}(w,c) + 1) / (\|D_c\| + \|V\|)$ |
| Log-version | $\hat{c} = \arg\max_c [\log P(c) + \sum \log P(w_i \| c)]$ |

### Conceptos Clave

**1. ¿Por qué se llama "Naive"?**
Porque asume independencia condicional entre características, lo cual es falso pero útil.

**2. ¿Cuándo funciona mejor Naive Bayes?**
- Datasets pequeños a medianos
- Texto corto (tweets, reseñas)
- Clases bien separadas por vocabulario

**3. ¿Qué es el problema del cero?**
Cuando una palabra del test no aparece en entrenamiento para una clase, el likelihood es 0, anulando todo el producto.

**4. ¿Por qué usar logaritmos?**
Para evitar underflow numérico al multiplicar muchas probabilidades pequeñas.

**5. ¿Cuál es la diferencia entre Multinomial y Bernoulli Naive Bayes?**
- Multinomial: usa frecuencias de palabras (Bag of Words con conteos)
- Bernoulli: usa solo presencia/ausencia de palabras (Booleano)

### Ventajas y Desventajas

| Ventajas | Desventajas |
|----------|-------------|
| Muy rápido de entrenar y clasificar | Suposición de independencia es falsa |
| Funciona bien con pocos datos | No captura relaciones entre palabras |
| Maneja alta dimensionalidad | Puede ser superado por modelos más complejos |
| Fácil de implementar | "No" + "bueno" = tratado como independiente |
| Probabilístico (da scores) | |

---

## Ejercicios Propuestos

### Ejercicio 1: Priors
Tienes 120 emails: 100 ham, 20 spam. Calcula los priors.

**Respuesta:**
- $P(\text{ham}) = 100/120 = 5/6 \approx 0.833$
- $P(\text{spam}) = 20/120 = 1/6 \approx 0.167$

### Ejercicio 2: Likelihoods
En emails spam, "gratis" aparece 40 veces. Total de tokens en spam: 800. Vocabulario: 400 palabras.

Calcula $P(\text{"gratis"} | \text{spam})$ con y sin Laplace.

**Respuesta:**
- Sin Laplace: $40/800 = 0.05$
- Con Laplace: $(40 + 1)/(800 + 400) = 41/1200 \approx 0.034$

### Ejercicio 3: Clasificación Completa
Dados:
- $P(\text{spam}) = 0.3$, $P(\text{ham}) = 0.7$
- Documento: "gratis oferta" (preprocesado)

Probabilidades (con Laplace ya aplicado):
- $P(\text{"gratis"}|\text{spam}) = 0.1$, $P(\text{"oferta"}|\text{spam}) = 0.05$
- $P(\text{"gratis"}|\text{ham}) = 0.01$, $P(\text{"oferta"}|\text{ham}) = 0.02$

Calcula la clase predicha.

**Solución:**

Spam:
$$
\log(0.3) + \log(0.1) + \log(0.05) = -1.204 + (-2.303) + (-2.996) = -6.503
$$

Ham:
$$
\log(0.7) + \log(0.01) + \log(0.02) = -0.357 + (-4.605) + (-3.912) = -8.874
$$

**Predicción**: Spam (-6.503 > -8.874)

### Ejercicio 4: Análisis Crítico
¿Por qué Naive Bayes podría fallar con la frase "no es bueno"?

**Respuesta**: Porque trata "no", "es" y "bueno" como independientes. No captura que "no" + "bueno" = negativo, mientras que "bueno" solo = positivo.

---

## Referencias para Estudio

- McCallum, A. & Nigam, K. "A Comparison of Event Models for Naive Bayes Text Classification" (1998)
- Manning, C. D. et al. *Introduction to Information Retrieval*, Capítulo 13
- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Capítulo 4
