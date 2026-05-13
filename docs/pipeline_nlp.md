# Pipeline de Procesamiento de Lenguaje Natural (PLN)

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
3. [Fases del Pipeline](#fases-del-pipeline)
4. [Requerimientos del Sistema](#requerimientos-del-sistema)
5. [Estructura del Proyecto](#estructura-del-proyecto)

---

## Introducción

El Procesamiento de Lenguaje Natural (PLN o NLP en inglés) es una rama de la inteligencia artificial que se encarga de la interacción entre las computadoras y el lenguaje humano. Este proyecto implementa un pipeline completo de PLN desde cero, cubriendo las etapas fundamentales: preprocesamiento, modelado de lenguaje, clasificación, evaluación y representación vectorial.

> **Objetivo**: Comprender los fundamentos matemáticos y algorítmicos de cada etapa del pipeline de PLN mediante implementaciones didácticas en Python.

---

## Arquitectura del Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Texto Crudo   │────▶│  Preprocesamiento│────▶│ Representación  │
│   (Raw Text)    │     │                 │     │   (TF-IDF/BoW)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                    ┌────────────────────────────────────┘
                    ▼
           ┌─────────────────┐     ┌─────────────────┐
           │   Modelado      │────▶│    Clasificación │
           │  (N-gramas /    │     │  (Naive Bayes)   │
           │  Word2Vec)      │     │                  │
           └─────────────────┘     └─────────────────┘
                    │                         │
                    └──────────┬──────────────┘
                               ▼
                      ┌─────────────────┐
                      │    Evaluación   │
                      │  (Métricas)     │
                      └─────────────────┘
```

---

## Fases del Pipeline

### 1. Preprocesamiento
Transforma el texto no estructurado en una forma normalizada y tokenizada lista para el análisis.

**Componentes principales:**
- **Normalización**: Conversión a minúsculas, eliminación de caracteres especiales
- **Tokenización**: División del texto en unidades léxicas (tokens)
- **Eliminación de Stopwords**: Filtrado de palabras funcionales de alta frecuencia y bajo contenido semántico
- **Stemming**: Reducción de palabras a su raíz morfológica

**Documentación detallada**: [preprocesamiento.md](preprocesamiento.md)

---

### 2. Representación Vectorial
Convierte texto en vectores numéricos que los algoritmos de machine learning pueden procesar.

**Técnicas implementadas:**
- **TF-IDF**: Ponderación de términos basada en frecuencia local y global
- **Word2Vec (Skip-gram)**: Embeddings densos aprendidos mediante redes neuronales superficiales

**Documentación detallada**: 
- [preprocesamiento.md](preprocesamiento.md) (TF-IDF)
- [word2vec.md](word2vec.md) (Word Embeddings)

---

### 3. Modelado de Lenguaje
Estima la probabilidad de secuencias de palabras para tareas de predicción y generación.

**Modelo implementado:**
- **Modelo de Bigramas**: Predicción de la siguiente palabra basada únicamente en la anterior
- **Suavizado de Laplace**: Técnica para manejar bigramas no observados en el entrenamiento

**Documentación detallada**: [n_gramas.md](n_gramas.md)

---

### 4. Clasificación
Asigna categorías o etiquetas a documentos de texto basándose en su contenido.

**Algoritmo implementado:**
- **Naive Bayes Multinomial**: Clasificador probabilístico basado en el teorema de Bayes con supuesto de independencia condicional entre características

**Documentación detallada**: [clasificacion.md](clasificacion.md)

---

### 5. Evaluación
Mide el rendimiento de los modelos mediante métricas estadísticas estandarizadas.

**Métricas implementadas:**
- **Matriz de Confusión**: Tabla de contingencia para visualizar aciertos y errores
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Capacidad de no etiquetar como positivo un negativo
- **Recall**: Capacidad de encontrar todos los positivos
- **F1-Score**: Media armónica entre precision y recall

**Documentación detallada**: [evaluacion.md](evaluacion.md)

---

## Requerimientos del Sistema

### Dependencias de Software
- **Python**: >= 3.10
- **NumPy**: Operaciones numéricas y álgebra lineal
- **Jupyter Notebook**: Entorno interactivo de desarrollo

### Instalación del Entorno
Este proyecto utiliza `uv` como gestor de paquetes y entornos virtuales.

```bash
# Crear entorno virtual (ya configurado en el proyecto)
uv venv

# Activar entorno
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate

# Instalar dependencias
uv pip install -r pyproject.toml
```

### Estructura de Datos de Entrada
Los algoritmos esperan diferentes formatos según la etapa:

| Etapa | Formato de Entrada | Ejemplo |
|-------|-------------------|---------|
| Preprocesamiento | `str` (texto crudo) | `"El gato come pescado"` |
| TF-IDF | `List[str]` (corpus) | `["doc1", "doc2", ...]` |
| Bigramas | `List[List[str]]` (tokens) | `[["yo", "como"], ...]` |
| Naive Bayes | `List[str]`, `List[str]` (docs, labels) | Ver `bayes/` |
| Word2Vec | `List[List[str]]` (corpus tokenizado) | `[["el", "perro", ...], ...]` |

---

## Estructura del Proyecto

```
.
├── bayes/
│   └── naive_bayes.ipynb          # Clasificador Naive Bayes
├── evaluacion/
│   └── evaluadores.ipynb          # Métricas de evaluación
├── n-gramas/
│   └── bigramas.ipynb             # Modelo de lenguaje bigramas
├── preprocesamiento/
│   └── preprocesamiento_basico.ipynb  # Preprocesamiento + TF-IDF
├── word2vec/
│   └── word_2_vec_skip_gram.ipynb # Word embeddings
├── docs/
│   ├── pipeline_nlp.md            # Este documento
│   ├── preprocesamiento.md        # Doc: Preprocesamiento
│   ├── n_gramas.md                # Doc: N-gramas
│   ├── clasificacion.md           # Doc: Naive Bayes
│   ├── evaluacion.md              # Doc: Métricas
│   └── word2vec.md                # Doc: Word2Vec
├── pyproject.toml                 # Configuración de dependencias
└── README.md                      # Índice principal
```

---

## Referencias

- Jurafsky, D. & Martin, J. H. *Speech and Language Processing* (3rd ed. draft)
- Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*
- Mikolov, T. et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
