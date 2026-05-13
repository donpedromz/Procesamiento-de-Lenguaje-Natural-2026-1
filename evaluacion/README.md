# Evaluación de Modelos de Clasificación

## Tabla de Contenidos
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Matriz de Confusión](#matriz-de-confusión)
3. [Accuracy (Exactitud)](#accuracy-exactitud)
4. [Precision (Precisión)](#precision-precisión)
5. [Recall (Sensibilidad / Exhaustividad)](#recall-sensibilidad)
6. [F1-Score](#f1-score)
7. [Relación entre Métricas](#relación-entre-métricas)
8. [Macro vs. Micro Average](#macro-vs-micro-average)
9. [Resumen para Examen](#resumen-para-examen)
10. [Ejercicios Propuestos](#ejercicios-propuestos)

---

## Introducción y Contexto

Una vez entrenado un clasificador, necesitamos **medir su rendimiento** de manera objetiva y estandarizada. La evaluación es crucial porque:

- Nos permite comparar diferentes modelos
- Identifica debilidades específicas (ej: confunde clase A con B)
- Guía la iteración y mejora del modelo
- Proporciona garantías antes de despliegue en producción

> **Evaluar un modelo solo con accuracy puede ser engañoso**, especialmente con datasets desbalanceados.

---

## Matriz de Confusión

### Definición

La **matriz de confusión** es una tabla de contingencia que visualiza el desempeño de un clasificador mostrando los aciertos y errores para cada clase.

### Estructura para Clasificación Binaria

|                | Predicción: Positivo | Predicción: Negativo |
|----------------|---------------------|---------------------|
| **Real: Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

### Términos Clave

| Símbolo | Nombre | Definición |
|---------|--------|------------|
| **VP** | Verdadero Positivo | Predije positivo y era positivo |
| **VN** | Verdadero Negativo | Predije negativo y era negativo |
| **FP** | Falso Positivo (Error Tipo I) | Predije positivo pero era negativo |
| **FN** | Falso Negativo (Error Tipo II) | Predije negativo pero era positivo |

> **Mnemotecnia**: El segundo término indica qué predije. VP = predije Positivo, era Verdadero.

### Ejemplo

Clasificador de spam (P = spam, N = ham):

|          | Pred: Spam | Pred: Ham |
|----------|-----------|-----------|
| Real: Spam | 80 (VP) | 20 (FN) |
| Real: Ham | 10 (FP) | 90 (VN) |

---

## Accuracy (Exactitud)

### Definición

Proporción de predicciones correctas sobre el total:

$$
\text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}
$$

### Ejemplo

$$
\text{Accuracy} = \frac{80 + 90}{80 + 90 + 10 + 20} = \frac{170}{200} = 0.85 \text{ (85%)}
$$

### Limitación Crítica

El accuracy puede ser **engañoso con clases desbalanceadas**.

**Ejemplo extremo:**
- Dataset: 95% negativos, 5% positivos
- Clasificador tonto: siempre predice negativo
- Accuracy = 95% ¡pero nunca detecta positivos!

> **Regla de oro**: Nunca uses accuracy como única métrica con datos desbalanceados.

---

## Precision (Precisión)

### Definición

De todas las instancias que predije como positivas, ¿cuántas realmente lo eran?

$$
\text{Precision} = \frac{VP}{VP + FP}
$$

### Interpretación

- **Alta precision**: Cuando digo positivo, casi siempre acierto
- **Baja precision**: Muchos falsos positivos (alarmas falsas)

### Ejemplo

Del ejemplo de spam:

$$
\text{Precision} = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.889 \text{ (88.9%)}
$$

### Casos de Uso donde Importa la Precision
- Detección de spam en email: prefiero que algunos spam pasen a que emails importantes vayan a spam
- Diagnóstico médico cuando el tratamiento es invasivo: no quiero tratar a sanos

---

## Recall (Sensibilidad)

### Definición

De todas las instancias realmente positivas, ¿cuántas detecté?

$$
\text{Recall} = \frac{VP}{VP + FN}
$$

### Interpretación

- **Alto recall**: Detecto casi todos los positivos
- **Bajo recall**: Me pierdo muchos positivos (falsos negativos)

### Ejemplo

$$
\text{Recall} = \frac{80}{80 + 20} = \frac{80}{100} = 0.80 \text{ (80%)}
$$

### Casos de Uso donde Importa el Recall
- Detección de cáncer: prefiero falsos positivos (más biopsias) a perder un caso real
- Detección de fraudes: quiero detectar la mayoría, aunque tenga falsas alarmas

---

## F1-Score

### El Dilema Precision vs. Recall

- Aumentar precision suele disminuir recall (y viceversa)
- Necesitamos una métrica que balancee ambos

### Definición

El **F1-Score** es la **media armónica** entre precision y recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Propiedades de la Media Armónica

- Penaliza desbalance entre precision y recall más severamente que la media aritmética
- Si precision = 0 o recall = 0, entonces F1 = 0
- F1 máximo = 1 (cuando precision = recall = 1)

### Ejemplo

$$
F1 = 2 \cdot \frac{0.889 \cdot 0.80}{0.889 + 0.80} = 2 \cdot \frac{0.711}{1.689} \approx 0.842
$$

### Fórmula Alternativa

$$
F1 = \frac{VP}{VP + \frac{FP + FN}{2}}
$$

> **Pregunta de examen**: ¿Por qué F1 usa media armónica y no aritmética?
> **Respuesta**: La media armónica penaliza más fuertemente cuando una de las dos métricas es muy baja, forzando un balance.

---

## Relación entre Métricas

### Escenarios Típicos

| Situación | Precision | Recall | Interpretación |
|-----------|-----------|--------|----------------|
| Predigo pocos, pero acierto | Alta | Baja | Conservador |
| Predijo muchos, algunos mal | Baja | Alta | Agresivo |
| Perfecto | 1.0 | 1.0 | Ideal |
| Terrible | Baja | Baja | Modelo inútil |

### Trade-off Precision-Recall

En muchos clasificadores, podemos ajustar un umbral de decisión:
- **Umbral alto**: pocos positivos predichos → alta precision, bajo recall
- **Umbral bajo**: muchos positivos predichos → baja precision, alto recall

---

## Macro vs. Micro Average

### Promedio Macro (Macro Average)

Calcula la métrica para cada clase y luego promedia:

$$
F1_{\text{macro}} = \frac{1}{|C|} \sum_{c \in C} F1(c)
$$

- Trata todas las clases igual
- Puede ser dominado por clases pequeñas
- Útil cuando todas las clases son igualmente importantes

### Promedio Micro (Micro Average)

Agrega los conteos de todas las clases y calcula una sola métrica:

$$
\text{Precision}_{\text{micro}} = \frac{\sum VP_c}{\sum VP_c + \sum FP_c}
$$

- Equivalente al accuracy en clasificación multiclase
- Favorece clases grandes
- Útil cuando las clases deben ponderarse por su frecuencia

### Promedio Ponderado (Weighted Average)

Promedio macro ponderado por el número de instancias de cada clase:

$$
F1_{\text{weighted}} = \sum_{c \in C} \frac{N_c}{N} \cdot F1(c)
$$

---

## Resumen para Examen

### Fórmulas a Memorizar

| Métrica | Fórmula | Cuándo usar |
|---------|---------|-------------|
| **Accuracy** | $\frac{VP + VN}{\text{Total}}$ | Clases balanceadas |
| **Precision** | $\frac{VP}{VP + FP}$ | Cuando FP son costosos |
| **Recall** | $\frac{VP}{VP + FN}$ | Cuando FN son costosos |
| **F1** | $2 \cdot \frac{P \cdot R}{P + R}$ | Balance P y R |

### Preguntas de Examen Frecuentes

**1. ¿Qué es un falso positivo?**
Predije positivo pero el real era negativo. Error Tipo I.

**2. ¿Por qué accuracy no sirve con datos desbalanceados?**
Porque un clasificador que siempre predice la clase mayoritaria tendrá alta accuracy sin ser útil.

**3. Si precision = 1.0 y recall = 0.5, ¿cuál es el F1?**
$$
F1 = 2 \cdot \frac{1.0 \cdot 0.5}{1.0 + 0.5} = 2 \cdot \frac{0.5}{1.5} = \frac{1}{1.5} \approx 0.667
$$

**4. ¿En qué caso preferirías alto recall sobre alta precision?**
En detección de enfermedades graves, donde es mejor tener falsas alarmas que perder casos reales.

**5. Dada una matriz de confusión, calcula todas las métricas.**
(Ejercicio clásico de examen)

### Relaciones Clave

- VP ↑ → Precision ↑ y Recall ↑
- FP ↑ → Precision ↓
- FN ↑ → Recall ↓
- VN no afecta Precision ni Recall (solo accuracy)

---

## Ejercicios Propuestos

### Ejercicio 1: Matriz 2x2
Clasificador de enfermedad:

|          | Pred: Enfermo | Pred: Sano |
|----------|--------------|-----------|
| Real: Enfermo | 45 | 5 |
| Real: Sano | 15 | 135 |

Calcula: Accuracy, Precision, Recall, F1.

**Solución:**
- VP = 45, VN = 135, FP = 15, FN = 5
- Accuracy = (45 + 135) / 200 = 180/200 = **0.90**
- Precision = 45 / (45 + 15) = 45/60 = **0.75**
- Recall = 45 / (45 + 5) = 45/50 = **0.90**
- F1 = 2 · (0.75 · 0.90) / (0.75 + 0.90) = 2 · 0.675 / 1.65 = **0.818**

### Ejercicio 2: Dataset Desbalanceado
Dataset: 990 negativos, 10 positivos.
Modelo predice siempre negativo.

**Pregunta**: ¿Cuál es el accuracy? ¿Es un buen modelo?

**Respuesta:**
- Accuracy = 990/1000 = **99%**
- VP = 0, FP = 0, FN = 10, VN = 990
- Precision = 0/0 (indefinido o 0)
- Recall = 0/10 = **0%**
- **No es buen modelo**: nunca detecta positivos.

### Ejercicio 3: Clasificación Multiclase
Dataset de 3 clases (A, B, C) con predicciones:

| Real \ Pred | A | B | C |
|-------------|---|---|---|
| A | 8 | 2 | 0 |
| B | 1 | 7 | 2 |
| C | 0 | 1 | 9 |

Calcula métricas por clase para A.

**Respuesta para clase A:**
- VP = 8
- FP (predije A pero no era A) = 1 + 0 = 1
- FN (era A pero no predije A) = 2 + 0 = 2
- Precision = 8 / (8 + 1) = **8/9 ≈ 0.889**
- Recall = 8 / (8 + 2) = **8/10 = 0.80**
- F1 = 2 · (0.889 · 0.80) / (0.889 + 0.80) ≈ **0.842**

### Ejercicio 4: Trade-off
Un modelo de detección de spam tiene precision = 0.95 y recall = 0.60. ¿Qué significa y cómo mejorar recall?

**Respuesta:**
- Significa que cuando marca spam, acierta 95% de las veces, pero solo detecta 60% del spam real.
- Para mejorar recall: bajar el umbral de decisión (aceptar más falsos positivos).

---

## Referencias para Estudio

- Powers, D. M. W. "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation" (2011)
- Manning, C. D. et al. *Introduction to Information Retrieval*, Capítulo 8
- Sokolova, M. & Lapalme, G. "A systematic analysis of performance measures for classification tasks" (2009)
