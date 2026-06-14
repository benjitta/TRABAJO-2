# -------------------------------------------------------------------
# Proyecto Data Science: Predicción Generación Solar (Atacama)
# Estudiantes: Jesús Cornejo, Benjamin Valverde, javier Delgado, Yorkshua Cerda
# -------------------------------------------------------------------

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings

# Sklearn & Keras (TODO: limpiar imports que no se estén usando al final)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
np.random.seed(42) # Fijar semilla para reproducibilidad

# --- FASE 1: EXTRACCIÓN Y PREPROCESAMIENTO ---

def fetch_nasa_power_data(anio_inicio=2019, anio_fin=2023, archivo_local='historico_solar.csv'):
    # Check rápido por si ya descargamos la data antes (ahorra tiempo de API)
    if os.path.exists(archivo_local):
        print(f"[OK] Cargando dataset local: {archivo_local}")
        df_raw = pd.read_csv(archivo_local)
        df_raw['Fecha_Hora'] = pd.to_datetime(df_raw['Fecha_Hora'])
        return df_raw

    dfs_anuales = []
    print(f"Iniciando request a NASA POWER ({anio_inicio}-{anio_fin})...")

    for year in range(anio_inicio, anio_fin + 1):
        print(f"Descargando data {year}...")
        
        # Formateo de fechas para la API
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M,RH2M,CLOUD_AMT",
            "community":  "RE",
            "longitude":  -68.2006,
            "latitude":   -22.9087,
            "start":      f"{year}0101",
            "end":        f"{year}1231",
            "format":     "JSON",
        }

        # Manejo de reintentos por si se cae la conexión
        for intento in range(3):
            try:
                res = requests.get(url, params=params, timeout=60)
                res.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"Fallo en intento {intento+1} para {year}: {e}")
                time.sleep(5)

        data = res.json()
        p_data = data['properties']['parameter']

        # Armar dataframes temporales
        df_rad = pd.DataFrame(list(p_data['ALLSKY_SFC_SW_DWN'].items()), columns=['Fecha_Hora', 'Radiacion_W_m2'])
        df_temp = pd.DataFrame(list(p_data['T2M'].items()), columns=['Fecha_Hora', 'Temperatura_C'])
        df_viento = pd.DataFrame(list(p_data['WS10M'].items()), columns=['Fecha_Hora', 'Viento_m_s'])
        df_humedad = pd.DataFrame(list(p_data['RH2M'].items()), columns=['Fecha_Hora', 'Humedad_Rel_%'])
        df_nubes = pd.DataFrame(list(p_data['CLOUD_AMT'].items()), columns=['Fecha_Hora', 'Nubosidad_%'])

        # Merge sucesivo
        df_year = df_rad.copy()
        for df_extra in [df_temp, df_viento, df_humedad, df_nubes]:
            df_year = pd.merge(df_year, df_extra, on='Fecha_Hora')

        dfs_anuales.append(df_year)
        time.sleep(2) # delay para no saturar la API

    df_final = pd.concat(dfs_anuales, ignore_index=True)
    df_final['Fecha_Hora'] = pd.to_datetime(df_final['Fecha_Hora'], format='%Y%m%d%H')
    
    # Limpieza básica
    df_final = df_final.replace(-999.0, np.nan)
    # df_final.dropna(inplace=True) # Mejor imputar después para no perder horas enteras
    df_final = df_final.sort_values('Fecha_Hora').reset_index(drop=True)
    
    # Feature Engineering: Codificación Cíclica de la Hora
    df_final['hora'] = df_final['Fecha_Hora'].dt.hour
    df_final['hora_sin'] = np.sin(2 * np.pi * df_final['hora'] / 24)
    df_final['hora_cos'] = np.cos(2 * np.pi * df_final['hora'] / 24)
    
    df_final.to_csv(archivo_local, index=False)
    print("Descarga finalizada y guardada.")
    return df_final


def calcular_generacion(df):
    """ Calcula MW usando fórmula física base + ruido blanco """
    base_mw = (df['Radiacion_W_m2'] * 150000 * 0.21 * 0.75) / 1_000_000
    # clipping en 0 porque no podemos generar energía negativa
    return (base_mw + np.random.normal(0, 15.0, len(df))).clip(lower=0)


# --- FASE 2: EDA (Análisis Exploratorio) ---

def plot_eda_basico(df):
    print("\n[Generando gráficos EDA...]")
    sns.set_theme(style="whitegrid")
    cols_numericas = ['Radiacion_W_m2', 'Temperatura_C', 'Viento_m_s', 'Humedad_Rel_%', 'Nubosidad_%', 'Generacion_MW']
    
    # Distribuciones
    fig, axes = plt.subplots(2, len(cols_numericas), figsize=(20, 8))
    plt.suptitle('Distribuciones y Outliers', fontsize=14)
    
    for i, col in enumerate(cols_numericas):
        sns.histplot(df[col], kde=True, ax=axes[0, i], color='#1f77b4')
        axes[0, i].set_title(col, fontsize=10)
        sns.boxplot(y=df[col], ax=axes[1, i], color='#2ca02c')
        axes[1, i].set_ylabel('')
        
    plt.tight_layout()
    plt.show()

    # Matriz de correlación y Scatters
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = plt.subplot(1, 4, 1)
    corr = df[cols_numericas].corr()
    # Solo mostramos la correlación contra el target
    sns.heatmap(corr[['Generacion_MW']].sort_values(by='Generacion_MW', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1, cbar=False)
    ax1.set_title('Corr vs Generación')

    # Scatter plots rápidos
    colors = ['darkorange', 'firebrick', 'slategray']
    x_vars = ['Radiacion_W_m2', 'Temperatura_C', 'Nubosidad_%']
    
    for idx, (var, color) in enumerate(zip(x_vars, colors), 2):
        ax = plt.subplot(1, 4, idx)
        sns.scatterplot(data=df.sample(2000), x=var, y='Generacion_MW', alpha=0.4, color=color, ax=ax)
        # plt.title(f'{var}')
        
    plt.tight_layout()
    plt.show()
#SPLIT, IMPUTACIÓN Y OUTLIERS

def prepare_datasets(df_raw):
    df_train_val, df_test = train_test_split(df_raw, test_size=0.20, shuffle=False)
    df_train, df_val = train_test_split(df_train_val, test_size=0.20, shuffle=False)

    cols = ['Radiacion_W_m2', 'Temperatura_C', 'Viento_m_s', 'Humedad_Rel_%', 'Nubosidad_%']
    imputer = KNNImputer(n_neighbors=5)
    
    df_train.loc[:, cols] = imputer.fit_transform(df_train[cols])
    df_val.loc[:, cols] = imputer.transform(df_val[cols])
    df_test.loc[:, cols] = imputer.transform(df_test[cols])

    for df in [df_train, df_val, df_test]:
        df = df[df['Radiacion_W_m2'] > 0].reset_index(drop=True)
        df['Generacion_MW'] = crear_generacion(df)

    q1 = df_train['Generacion_MW'].quantile(0.25)
    q3 = df_train['Generacion_MW'].quantile(0.75)
    limite_sup = q3 + 1.5 * (q3 - q1)

    for df in [df_train, df_val, df_test]:
        df['Generacion_MW'] = np.where(df['Generacion_MW'] > limite_sup, limite_sup, df['Generacion_MW'])

    return df_train, df_val, df_test

def prepare_features(df_train, df_val, df_test):
    features = ['Radiacion_W_m2', 'Temperatura_C', 'Viento_m_s', 'Humedad_Rel_%', 'Nubosidad_%', 'hora_sin', 'hora_cos']
    return (df_train[features], df_train['Generacion_MW'], 
            df_val[features], df_val['Generacion_MW'], 
            df_test[features], df_test['Generacion_MW'])

def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    transformers = [('scaler', StandardScaler(), numeric_cols)]
    
    if categorical_cols:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder='drop')

#REGRESIÓN


def train_regression_models(X_train, y_train):
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    return lr, rf

def print_regression_results(models, X_test, y_test):
    predictions = {}
    for nombre, modelo in models:
        y_pred = modelo.predict(X_test)
        predictions[nombre] = y_pred
        print(f"\n{nombre}: R²={r2_score(y_test, y_pred):.4f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    return predictions

def plot_regression_comparison(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Real', alpha=0.8)
    for nombre, y_pred in predictions.items():
        plt.plot(y_pred, label=nombre, alpha=0.7)
    plt.legend()
    plt.show()

# SECCIÓN DE CLASIFICACIÓN
    
    def plot_class_distribution(y_train_c, y_test_c):
    labels = ['No Alto', 'Alto']
    
    #Calculamos las frecuencias absolutas para cada partición
    train_counts = np.bincount(y_train_c, minlength=2)
    test_counts  = np.bincount(y_test_c,  minlength=2)
    
    x = np.arange(len(labels))
    width = 0.35  # Ancho de las barras para evitar superposición

    plt.figure(figsize=(7, 4))
    
    # Graficamos ambas distribuciones lado a lado para facilitar la comparación
    plt.bar(x - width / 2, train_counts, width, label='Train (Original)')
    plt.bar(x + width / 2, test_counts,  width, label='Test')
    
    # Configuramos ejes y leyendas
    plt.xticks(x, labels)
    plt.ylabel('Cantidad de muestras')
    plt.title('Distribución de Clases (Antes de SMOTE)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix, title):
    #Genera un mapa de calor para la matriz de confusión para evaluar los Falsos Positivos y Falsos Negativos.
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Alto', 'Alto'])
    plt.yticks(tick_marks, ['No Alto', 'Alto'])
    
    # Calculamos un umbral dinámico para cambiar el color del texto y asegurar que sea legible sin importar el color de fondo de la celda.
    thresh = matrix.max() / 2.0
    
    # Iteramos sobre la matriz para anotar los valores exactos
    for i, j in np.ndindex(matrix.shape):
        color_texto = 'white' if matrix[i, j] > thresh else 'black'
        plt.text(j, i, format(matrix[i, j], 'd'),
                 ha='center', va='center',
                 color=color_texto)
                 
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curve(y_true, y_proba, title):
    
    #Renderiza la Curva Precision-Recall (PR) y la Curva ROC esto nos da una mejor perspectiva cuando existe desbalanceo severo, mientras que la ROC evalúa la capacidad de discriminación general del modelo.
    # Cálculos para la curva Precision-Recall
    ap = average_precision_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    # Cálculos para la curva ROC
    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(10, 4))
    
    # Subplot 1: Curva PR
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR AUC = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall — {title}')
    plt.legend()

    # Subplot 2: Curva ROC
    plt.subplot(1, 2, 2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatorio')
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC — {title}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test_c):
    #Binariza la variable objetivo, aplica SMOTE en el conjunto de entrenamiento y entrena los estimadores clásicos definido luego compara su rendimiento mediante reporte de métricas y gráficos.
    #Definimos como clase 'Alto' a las muestras sobre el percentil 75
    threshold = np.quantile(y_train, 0.75)
    y_train_c = np.where(y_train > threshold, 1, 0)
    
    # Revisión visual del desbalanceo original
    plot_class_distribution(y_train_c, y_test_c)

    # Inicializamos SMOTE asegurando reproducibilidad
    # fit_resample SOLO se aplica a X_train para no contaminar el test
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train_c)

    # Diccionario de clasificadores tradicionales a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    print('\n' + '=' * 55)
    print('CLASIFICACIÓN — EVALUACIÓN INDIVIDUAL')
    print('=' * 55)

    mejor_auc = 0
    mejor_modelo = ""

    # Ciclo de entrenamiento y evaluación por cada algoritmo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_smote, y_train_smote)
        
        # Predicciones de clase dura y probabilidades (necesarias para ROC/PR)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        print(f"\nReporte de Clasificación: {nombre}")
        print(classification_report(y_test_c, y_pred))
        
        #Identificamos dinámicamente cuál modelo discrimina mejor (basado en ROC-AUC)
        roc_auc = roc_auc_score(y_test_c, y_proba)
        if roc_auc > mejor_auc:
            mejor_auc = roc_auc
            mejor_modelo = nombre
            
        # Generación de visualizaciones por modelo
        plot_confusion_matrix(confusion_matrix(y_test_c, y_pred), f'Matriz Confusión — {nombre}')
        plot_roc_pr_curve(y_test_c, y_proba, nombre)
        
    print(f"\n>> El mejor modelo clásico es: {mejor_modelo} (ROC-AUC: {mejor_auc:.4f})")
    
    return threshold, X_train_smote, y_train_smote
    
# DEEP LEARNING / MLP BINARIO

def build_deep_learning_model(input_shape):
    # Esto construye la arquitectura de la red neuronal e Incorpora inicialización He, el Batch Normalization es para estabilizar el aprendizaje y Dropout para prevenir el sobreajuste.
    model = Sequential([
        # Primera capa oculta
        Dense(64, activation='relu', kernel_initializer='he_uniform', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Segunda capa oculta
        Dense(32, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa de salida con Sigmoid para clasificación binaria
        Dense(1, activation='sigmoid'),
    ])
    
    #Compilación estándar para clasificación probabilística
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def plot_training_history_dl(history):
    # Esto grafica la evolución de la función de pérdida y la exactitud por época, sirve para ver si hay underfitting o la aparición de overfitting.
    plt.figure(figsize=(10, 4))
    
    # Gráfico de la Función de Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Entrenamiento MLP — Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    # Gráfico de la Métrica (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Entrenamiento MLP — Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def evaluate_dl_model(modelo_dl, X_test_pre, y_test_c):
    loss, acc = modelo_dl.evaluate(X_test_pre, y_test_c, verbose=0)
    
    print('\n' + '=' * 55)
    print('DEEP LEARNING MLP — RESULTADOS')
    print('=' * 55)
    print(f'  Loss en test:     {loss:.4f}')
    print(f'  Accuracy en test: {acc:.4f}')

    # Extraemos probabilidades y las binarizamos al umbral de 0.5
    y_proba_dl = modelo_dl.predict(X_test_pre).flatten()
    y_pred_dl  = (y_proba_dl > 0.5).astype(int)

    print('\nReporte de clasificación (DL):')
    print(classification_report(y_test_c, y_pred_dl))
    
    # Visualizaciones finales para la red neuronal
    matriz_resultante = confusion_matrix(y_test_c, y_pred_dl)
    plot_confusion_matrix(matriz_resultante, 'Matriz de confusión — Deep Learning')
    plot_roc_pr_curve(y_test_c, y_proba_dl, 'Deep Learning')


def main():
    df_raw = fetch_nasa_power_data(anio_inicio=2019, anio_fin=2023)
    df_train, df_val, df_test = prepare_datasets(df_raw)
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(df_train, df_val, df_test)

    preprocessor = build_preprocessor(X_train)
    X_train_pre  = preprocessor.fit_transform(X_train)
    X_val_pre    = preprocessor.transform(X_val)
    X_test_pre   = preprocessor.transform(X_test)

    # Regresión
    modelo_lineal, rf = train_regression_models(X_train_pre, y_train)
    predictions = print_regression_results([("Regresión Lineal", modelo_lineal), ("Random Forest", rf)], X_test_pre, y_test)
    plot_regression_comparison(y_test, predictions)

    # Clasificación (Logística vs KNN)
    threshold, X_train_smote, y_train_smote = train_and_evaluate_classifiers(
        X_train_pre, y_train, X_test_pre, np.where(y_test > np.quantile(y_train, 0.75), 1, 0)
    )
    y_test_c = np.where(y_test > threshold, 1, 0)
    y_val_c  = np.where(y_val  > threshold, 1, 0)

    # Deep Learning
    modelo_dl = build_deep_learning_model(X_train_pre.shape[1])
    print('\n' + '=' * 55)
    print('DEEP LEARNING — RESUMEN DEL MODELO')
    print('=' * 55)
    modelo_dl.summary()

    history = modelo_dl.fit(
        X_train_smote, y_train_smote,
        epochs=30, batch_size=32,
        validation_data=(X_val_pre, y_val_c),   
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ],
        verbose=1,
    )

    plot_training_history_dl(history)
    evaluate_dl_model(modelo_dl, X_test_pre, y_test_c)


if _name_ == '_main_':
    main()
