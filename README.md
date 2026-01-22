# Análisis de Sentimientos en Reseñas de Películas

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.2-09a3d5.svg)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Proyecto de clasificación binaria de sentimientos (positivo/negativo) en reseñas de películas del dataset IMDB. Implementa y compara dos arquitecturas: una clásica basada en TF-IDF y otra basada en embeddings semánticos pre-entrenados.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitecturas Implementadas](#-arquitecturas-implementadas)
- [Resultados](#-resultados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Análisis de Resultados](#-análisis-de-resultados)
- [Limitaciones](#-limitaciones)
- [Próximos Pasos](#-próximos-pasos)
- [Referencias](#-referencias)

## 🎯 Descripción del Proyecto

Este proyecto desarrolla y evalúa dos enfoques diferentes para clasificación de sentimientos en texto:

1. **Enfoque clásico**: TF-IDF + Logistic Regression
2. **Enfoque basado en embeddings**: tok2vec (spaCy) + Linear SVM

El objetivo es comparar ambas metodologías en términos de rendimiento, interpretabilidad y eficiencia computacional, demostrando competencias en:
- Preprocesamiento de texto con spaCy
- Feature engineering (TF-IDF, embeddings)
- Evaluación rigurosa de modelos
- Análisis de errores e interpretabilidad

## 🏗️ Arquitecturas Implementadas

### Arquitectura 1: TF-IDF + Logistic Regression

**Pipeline:**
```
Texto → Preprocesamiento spaCy → TF-IDF Vectorization → Logistic Regression → Predicción
```

**Características:**
- **Vectorización**: TF-IDF con 5,000 features máximas, unigramas y bigramas
- **Clasificador**: Logistic Regression con regularización L2 (C=1.0)
- **Ventajas**: Alta interpretabilidad, rápido entrenamiento, bajo uso de memoria
- **Desventajas**: No captura similitud semántica entre palabras

### Arquitectura 2: tok2vec + Linear SVM

**Pipeline:**
```
Texto → Preprocesamiento spaCy → tok2vec Embeddings → Linear SVM → Predicción
```

**Características:**
- **Vectorización**: Embeddings pre-entrenados de spaCy `en_core_web_lg` (300 dims)
- **Clasificador**: Linear SVM con kernel lineal (C=1.0)
- **Ventajas**: Captura relaciones semánticas, vectores densos pre-entrenados
- **Desventajas**: Menos interpretable, mayor costo computacional

## 📊 Resultados

### Métricas de Evaluación

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **LR + TF-IDF** | 0.8750 | 0.8842 | 0.8640 | 0.8740 | 0.9445 |
| **SVM + tok2vec** | 0.8590 | 0.8452 | 0.8800 | 0.8623 | 0.9312 |
| **Baseline (mayoría)** | 0.5000 | - | - | - | 0.5000 |

**Observaciones clave:**
- Ambos modelos superan significativamente el baseline
- **TF-IDF + LR** obtiene mejor F1-Score general (0.8740 vs 0.8623)
- **tok2vec + SVM** tiene mejor recall (0.8800 vs 0.8640) → detecta más casos positivos
- Validación cruzada 5-fold confirma estabilidad de los modelos

### Visualizaciones

<p align="center">
  <img src="results/04_confusion_matrices.png" width="800" alt="Matrices de Confusión">
</p>

<p align="center">
  <img src="results/05_roc_curves.png" width="600" alt="Curvas ROC">
</p>

<p align="center">
  <img src="results/08_feature_importance.png" width="800" alt="Features Importantes">
</p>

## 📁 Estructura del Proyecto

```
sentiment-analysis-nlp/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── sentiment_analysis_improved.ipynb  # Notebook principal con análisis completo
├── movie_reviews_dataset_5000.csv     # Dataset de reseñas (no incluido en repo)
│
├── models/                            # Modelos entrenados (generados al ejecutar)
│   ├── lr_tfidf_sentiment.pkl
│   ├── svm_tok2vec_sentiment.pkl
│   ├── tfidf_vectorizer.pkl
│   └── spacy_model_info.txt
│
├── results/                           # Visualizaciones y métricas (generadas al ejecutar)
│   ├── model_metrics.csv
│   ├── 01_sentiment_distribution.png
│   ├── 02_length_distribution.png
│   ├── 03_top_words_raw.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curves.png
│   ├── 06_metrics_comparison.png
│   ├── 07_confidence_analysis.png
│   ├── 08_feature_importance.png
│   └── 09_learning_curves.png
│
└── .gitignore                         # Archivos ignorados por Git
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip para gestión de paquetes

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar modelo de spaCy**
```bash
python -m spacy download en_core_web_lg
```

## 💻 Uso

### Opción 1: Google Colab (Recomendado)

1. Abre el notebook en Google Colab
2. Sube el archivo `movie_reviews_dataset_5000.csv` cuando se te solicite
3. Ejecuta las celdas secuencialmente

### Opción 2: Jupyter Notebook Local

```bash
jupyter notebook sentiment_analysis_improved.ipynb
```

### Opción 3: Usar Modelos Pre-entrenados

```python
import joblib
import spacy

# Cargar modelos guardados
lr_model = joblib.load('models/lr_tfidf_sentiment.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
nlp = spacy.load('en_core_web_lg')

# Función de preprocesamiento (copiar del notebook)
def preprocess_text(text):
    # ... código de preprocesamiento
    pass

# Predecir sentimiento
new_review = "This movie was absolutely fantastic!"
clean_text = preprocess_text(new_review)
text_tfidf = tfidf.transform([clean_text])
prediction = lr_model.predict(text_tfidf)[0]
probability = lr_model.predict_proba(text_tfidf)[0]

print(f"Sentimiento: {'Positivo' if prediction == 1 else 'Negativo'}")
print(f"Confianza: {probability[prediction]*100:.2f}%")
```

## 🔬 Metodología

### 1. Preprocesamiento de Texto

Pipeline de preprocesamiento con spaCy (`en_core_web_lg`):

1. **Limpieza**: Eliminación de HTML tags, URLs, caracteres especiales
2. **Normalización**: Conversión a minúsculas
3. **Lematización**: Reducción de palabras a su forma base (e.g., "running" → "run")
4. **Filtrado de stopwords**: Eliminación de palabras comunes excepto **negaciones** (not, never, no)
5. **Filtrado de tokens**: Solo tokens alfabéticos de >1 carácter

**Decisión crítica**: Preservar negaciones porque son esenciales para análisis de sentimientos
- Ejemplo: "not good" tiene significado opuesto a "good"

### 2. Feature Engineering

#### TF-IDF (Term Frequency-Inverse Document Frequency)

- **max_features=5000**: Vocabulario limitado a 5,000 términos más frecuentes
- **ngram_range=(1,2)**: Unigramas y bigramas para capturar frases
- **min_df=2**: Ignora términos que aparecen en <2 documentos
- **sublinear_tf=True**: Aplica escala logarítmica a frecuencias

#### tok2vec Embeddings

- **Modelo**: spaCy `en_core_web_lg` (300 dimensiones)
- **Estrategia**: Promedio de vectores de todos los tokens del documento
- **Ventaja**: Captura similitud semántica pre-aprendida en corpus masivo

### 3. Evaluación

**Estrategia de validación:**
- Train/Test split: 80/20 estratificado
- Validación cruzada: 5-fold StratifiedKFold
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC

**Análisis realizado:**
- Matrices de confusión (conteos y porcentajes)
- Curvas ROC y AUC
- Análisis de errores (falsos positivos y falsos negativos)
- Análisis de confianza en predicciones
- Features más discriminativas (TF-IDF)
- Curvas de aprendizaje

## 📈 Análisis de Resultados

### Comparación de Arquitecturas

**TF-IDF + Logistic Regression:**
- ✅ **Mejor F1-Score general** (0.8740)
- ✅ **Alta interpretabilidad**: Podemos ver qué palabras influyen más
- ✅ **Rápido**: Entrenamiento e inferencia muy eficientes
- ✅ **Bajo uso de memoria**: Matrices sparse
- ❌ No captura similitud semántica ("excellent" y "great" son palabras independientes)

**tok2vec + Linear SVM:**
- ✅ **Mejor recall** (0.8800): Detecta más casos positivos
- ✅ **Representación semántica**: Palabras similares tienen vectores similares
- ✅ **Pre-entrenado**: Aprovecha conocimiento de corpus masivo
- ❌ Menor interpretabilidad
- ❌ Mayor costo computacional
- ❌ Requiere más memoria (vectores densos)

### Features Más Importantes (TF-IDF)

**Top términos que indican sentimiento POSITIVO:**
- excellent, perfect, wonderful, brilliant, outstanding
- best, great, amazing, loved, masterpiece

**Top términos que indican sentimiento NEGATIVO:**
- waste, awful, boring, terrible, worst
- bad, poor, disappoint, dull, stupid

**Observación**: El modelo captura correctamente palabras con fuerte carga emocional.

### Análisis de Errores

**Casos difíciles para ambos modelos:**
- Reseñas con sentimientos mixtos: "Acting was good but plot was boring"
- Sarcasmo e ironía: "Oh great, another terrible sequel"
- Reseñas neutrales: "Not great, not terrible, just average"
- Sentimientos contextuales que requieren comprensión profunda

**Patrón identificado**: El modelo tiene menor confianza en predicciones erróneas vs correctas, indicando que "sabe cuando no sabe".

## ⚠️ Limitaciones

1. **Dataset balanceado**: 50/50 positivo/negativo. Performance en datos reales desbalanceados puede variar.

2. **Clasificación binaria**: No captura intensidad (muy positivo vs ligeramente positivo) ni neutralidad.

3. **Dominio específico**: Entrenado en reseñas de películas. Rendimiento en otros dominios (productos, restaurantes) requiere validación.

4. **Contexto limitado**: No maneja efectivamente sarcasmo, ironía o referencias culturales complejas.

5. **Embeddings estáticos**: tok2vec no es contextual (vs BERT que genera embeddings dependientes del contexto).

## 🚀 Próximos Pasos

### Mejoras de Corto Plazo

1. **Modelos basados en Transformers**: Fine-tuning de BERT, RoBERTa o DistilBERT
2. **Ensemble de modelos**: Combinar TF-IDF + LR y tok2vec + SVM
3. **Optimización de hiperparámetros**: GridSearchCV o búsqueda bayesiana
4. **Evaluación multi-dominio**: Testear en otros tipos de reseñas

### Mejoras de Largo Plazo

1. **Clasificación multi-clase**: Muy negativo, negativo, neutral, positivo, muy positivo
2. **Aspect-Based Sentiment Analysis**: Analizar sentimientos sobre aspectos específicos
3. **Detección de sarcasmo e ironía**: Modelos especializados
4. **Pipeline de producción**: API REST, containerización, monitoreo

## 📚 Referencias

### Dataset
- **IMDB Movie Reviews Dataset**: 5,000 reseñas balanceadas de películas

### Librerías y Frameworks
- **scikit-learn**: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*
- **spaCy**: Honnibal & Montani (2017). *spaCy 2: Natural language understanding with Bloom embeddings*

### Papers Relevantes
- Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL.
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
- Pang & Lee (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval.

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
