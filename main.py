# Proyecto Data Science - Prediccion Generacion Solar Atacama
# Integrantes: Jesus Cornejo, Benjamin Valverde, Javier Delgado, Yorkshua Cerda
#https://canva.link/qpgrh1gr6f2c5mf enlace de la presentacion

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Apaga el mensaje molesto de TensorFlow

import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42) # Semilla fija para reproducibilidad


# ==============================================================
# EXTRACCION DE DATOS Y LIMPIEZA
# ==============================================================
def obtener_datos_nasa(archivo_destino='historico_solar.csv'):
    """Obtiene datos del clima; usa cache si ya se descargo. Incluye manejo de reintentos."""
    if os.path.exists(archivo_destino):
        print("CSV local encontrado. Cargando datos...")
        df_local = pd.read_csv(archivo_destino)
        df_local['Fecha_Hora'] = pd.to_datetime(df_local['Fecha_Hora'])
        return df_local

    print("No hay CSV local. Consultando API de la NASA (esto tomara unos segundos)...")
    data_por_anio = []
    
    cabeceras = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for anio in range(2019, 2024):
        print(f"-> Extrayendo registros del {anio}")
        url_api = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        parametros = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M,RH2M,CLOUD_AMT",
            "community":  "RE",
            "longitude":  -68.2006,
            "latitude":   -22.9087,
            "start":      f"{anio}0101",
            "end":        f"{anio}1231",
            "format":     "JSON",
        }

        exito = False
        for intento in range(3):
            try:
                respuesta = requests.get(url_api, params=parametros, headers=cabeceras, timeout=40)
                respuesta.raise_for_status() 
                datos_json = respuesta.json()['properties']['parameter']

                df_temporal = pd.DataFrame(list(datos_json['ALLSKY_SFC_SW_DWN'].items()), columns=['Fecha_Hora', 'Radiacion'])
                df_temporal['Temperatura'] = list(datos_json['T2M'].values())
                df_temporal['Viento'] = list(datos_json['WS10M'].values())
                df_temporal['Humedad'] = list(datos_json['RH2M'].values())
                df_temporal['Nubosidad'] = list(datos_json['CLOUD_AMT'].values())

                data_por_anio.append(df_temporal)
                exito = True
                
                time.sleep(2) 
                break 

            except requests.exceptions.RequestException as error_red:
                print(f"   [Aviso] Fallo el intento {intento + 1}/3 para el año {anio}: Esperando para reintentar...")
                time.sleep(4) 
        
        if not exito:
            print(f"Error critico: No se pudo descargar el año {anio} despues de 3 intentos.")
            return None

    df_final = pd.concat(data_por_anio, ignore_index=True)
    df_final['Fecha_Hora'] = pd.to_datetime(df_final['Fecha_Hora'], format='%Y%m%d%H')
    df_final = df_final.replace(-999.0, np.nan)
    df_final.to_csv(archivo_destino, index=False)
    
    return df_final


def ingenieria_y_limpieza(df):
    """Aplica feature engineering, limpia noches y escala."""
    print("\nAplicando ingenieria de variables y tratando nulos...")

    df['hora'] = df['Fecha_Hora'].dt.hour
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

    df['Generacion_MW'] = (df['Radiacion'] * 150000 * 0.21 * 0.75) / 1000000
    df['Generacion_MW'] = df['Generacion_MW'] + np.random.normal(0, 15.0, len(df))
    df['Generacion_MW'] = df['Generacion_MW'].clip(lower=0)

    df_dia = df[df['Radiacion'] > 0].reset_index(drop=True)

    columnas_features = ['Radiacion', 'Temperatura', 'Viento', 'Humedad', 'Nubosidad', 'hora_sin', 'hora_cos']
    caracteristicas = df_dia[columnas_features]
    target = df_dia['Generacion_MW']

    feat_train, feat_test, tgt_train, tgt_test = train_test_split(
        caracteristicas, target, test_size=0.2, random_state=42, shuffle=False
    )

    imputador = KNNImputer(n_neighbors=5)
    feat_train_imp = pd.DataFrame(imputador.fit_transform(feat_train), columns=columnas_features)
    feat_test_imp = pd.DataFrame(imputador.transform(feat_test), columns=columnas_features)

    q1 = tgt_train.quantile(0.25)
    q3 = tgt_train.quantile(0.75)
    limite_superior = q3 + 1.5 * (q3 - q1)

    tgt_train = tgt_train.clip(upper=limite_superior)
    tgt_test = tgt_test.clip(upper=limite_superior)

    escalador = StandardScaler()
    feat_train_esc = escalador.fit_transform(feat_train_imp)
    feat_test_esc = escalador.transform(feat_test_imp)

    return feat_train_esc, feat_test_esc, tgt_train, tgt_test, feat_train_imp


# ==============================================================
# VISUALIZACIONES EDA 
# ==============================================================
def graficos_eda_semana3(df_entrenamiento, target):
    """Muestra distribuciones, outliers y dispersion segun la Semana 3."""
    print("Levantando graficos de Exploracion de Datos (EDA)...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Histogramas y Boxplots para ver distribucion y outliers
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4))
    
    sns.histplot(target, kde=True, ax=ejes[0], color='#3498db')
    ejes[0].set_title('Distribucion de Generacion (MW)')
    
    sns.boxplot(y=target, ax=ejes[1], color='#2ecc71')
    ejes[1].set_title('Boxplot Generacion (Outliers tratados)')
    
    # Radiacion vs Generacion (Scatterplot)
    sns.scatterplot(x=df_entrenamiento['Radiacion'], y=target, ax=ejes[2], alpha=0.3, color='#e74c3c')
    ejes[2].set_title('Dispersion: Radiacion vs Generacion')
    
    plt.tight_layout()
    plt.show()

def grafico_correlacion_triangular(df_entrenamiento, target):
    """Semana 3: Matriz de correlacion personalizada (mitad inferior)."""
    datos_completos = df_entrenamiento.copy()
    datos_completos['Generacion_MW'] = target.values
    
    matriz_corr = datos_completos.corr()
    mascara_triangulo = np.triu(np.ones_like(matriz_corr, dtype=bool))
    
    fig, eje = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz_corr, mask=mascara_triangulo, cmap='YlOrRd', 
                annot=True, fmt=".2f", linewidths=1.5, ax=eje, 
                cbar_kws={"shrink": .85})
    
    eje.set_title('Correlaciones (Mitad Inferior) - Atacama Solar', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


# ==============================================================
# EVALUACION DE MODELOS 
# ==============================================================
def evaluar_modelos(feat_train, feat_test, tgt_train, tgt_test):
    print("\n--- Modelos de Regresion ---")

    lin_reg = LinearRegression()
    lin_reg.fit(feat_train, tgt_train)
    preds_lr = lin_reg.predict(feat_test)

    bosque = RandomForestRegressor(n_estimators=115, max_depth=6, random_state=42)
    bosque.fit(feat_train, tgt_train)
    preds_rf = bosque.predict(feat_test)

    print(f"Regresion Lineal -> R2: {r2_score(tgt_test, preds_lr):.3f} | RMSE: {np.sqrt(mean_squared_error(tgt_test, preds_lr)):.3f}")
    print(f"Random Forest    -> R2: {r2_score(tgt_test, preds_rf):.3f} | RMSE: {np.sqrt(mean_squared_error(tgt_test, preds_rf)):.3f}")

    # GRAFICO: Comparacion Regresion Lineal vs Random Forest 
    plt.figure(figsize=(12, 4))
    # Mostramos solo una muestra de 150 puntos para que el grafico se entienda
    plt.plot(tgt_test.values[:150], label='Real', color='black', alpha=0.8, linewidth=2)
    plt.plot(preds_lr[:150], label='Pred. Regresion Lineal', color='orange', linestyle='--', alpha=0.7)
    plt.plot(preds_rf[:150], label='Pred. Random Forest', color='green', alpha=0.7)
    plt.title('Comparacion Predictiva: Modelos de Regresion (150 muestras)')
    plt.ylabel('Generacion (MW)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Clasificacion (Target > P75) ---")
    corte_alta = tgt_train.quantile(0.75)
    labels_train = (tgt_train > corte_alta).astype(int)
    labels_test = (tgt_test > corte_alta).astype(int)

    balanceador = SMOTE(random_state=42)
    feat_train_bal, labels_train_bal = balanceador.fit_resample(feat_train, labels_train)

    clasificador_knn = KNeighborsClassifier(n_neighbors=7)
    clasificador_knn.fit(feat_train_bal, labels_train_bal)
    preds_knn = clasificador_knn.predict(feat_test)

    print("Reporte K-Vecinos (balanceado con SMOTE):")
    print(classification_report(labels_test, preds_knn))

    # GRAFICO: Matriz de confusion de la Clasificacion 
    cm = confusion_matrix(labels_test, preds_knn)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['No Alto (0)', 'Alto (1)'], yticklabels=['No Alto (0)', 'Alto (1)'])
    plt.title('Matriz de Confusion - KNN Classifier')
    plt.ylabel('Valor Real')
    plt.xlabel('Prediccion')
    plt.tight_layout()
    plt.show()

    return feat_train_bal, labels_train_bal, labels_test
 

# ==============================================================
# RED NEURONAL 
# ==============================================================
def entrenar_red_neuronal(feat_train, labels_train, feat_test, labels_test):
    print("\n--- Entrenando Red Neuronal ---")

    red_solar = Sequential()
    red_solar.add(Dense(54, activation='relu', input_shape=(feat_train.shape[1],)))
    red_solar.add(Dropout(0.35))
    red_solar.add(Dense(22, activation='relu'))
    red_solar.add(Dropout(0.15))
    red_solar.add(Dense(1, activation='sigmoid'))

    optimizador_adam = Adam(learning_rate=0.0015)
    red_solar.compile(optimizer=optimizador_adam, loss='binary_crossentropy', metrics=['accuracy'])

    freno = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    historial = red_solar.fit(
        feat_train, labels_train,
        epochs=35,
        batch_size=32,
        validation_data=(feat_test, labels_test),
        callbacks=[freno],
        verbose=0
    )

    perdida, precision = red_solar.evaluate(feat_test, labels_test, verbose=0)
    print(f"Resultados Finales NN -> Loss: {perdida:.4f} | Accuracy: {precision:.4f}")

    # Graficos de entrenamiento de la Red Neuronal
    plt.style.use('ggplot')
    fig, (graf_loss, graf_acc) = plt.subplots(1, 2, figsize=(12, 5))
    
    graf_loss.plot(historial.history['loss'], color='navy', label='Fallo en Train', linewidth=2)
    graf_loss.plot(historial.history['val_loss'], color='crimson', label='Fallo en Validacion', linestyle='-.')
    graf_loss.set_title('Evolucion del Error (Loss)')
    graf_loss.legend()
    
    graf_acc.plot(historial.history['accuracy'], color='forestgreen', label='Aciertos Train')
    graf_acc.plot(historial.history['val_accuracy'], color='darkorange', label='Aciertos Validacion')
    graf_acc.axhline(y=0.85, color='black', linestyle=':', label='Meta 85%')
    graf_acc.set_title('Evolucion de la Precision')
    graf_acc.legend()
    
    plt.tight_layout()
    plt.show()


# ==============================================================
# FLUJO PRINCIPAL
# ==============================================================
if __name__ == '__main__':
    print("Iniciando pipeline Atacama Solar...")

    df_clima = obtener_datos_nasa()

    if df_clima is not None:
        train_feat_sc, test_feat_sc, train_tgt_limpio, test_tgt_limpio, df_imputado = ingenieria_y_limpieza(df_clima)

        # Agregamos la funcion de los graficos EDA de la Semana 3
        graficos_eda_semana3(df_imputado, train_tgt_limpio)
        grafico_correlacion_triangular(df_imputado, train_tgt_limpio)

        train_feat_bal, train_labels_bal, test_labels_clasif = evaluar_modelos(
            train_feat_sc, test_feat_sc, train_tgt_limpio, test_tgt_limpio
        )

        entrenar_red_neuronal(train_feat_bal, train_labels_bal, test_feat_sc, test_labels_clasif)

        print("\nPipeline finalizado con exito. Todos los graficos fueron generados.")
    else:
        print("\nError fatal: No se pudieron obtener los datos necesarios para el proyecto. Revisar conexion o API de la NASA.")
