
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
