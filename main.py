# CLASIFICACIÓN (MANTENIENDO GRÁFICOS ORIGINALES)
def plot_class_distribution(y_train_c, y_test_c):
    """Gráfico de barras original restaurado"""
    labels       = ['No Alto', 'Alto']
    train_counts = np.bincount(y_train_c, minlength=2)
    test_counts  = np.bincount(y_test_c,  minlength=2)
    x     = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, train_counts, width, label='Train (Original)')
    plt.bar(x + width / 2, test_counts,  width, label='Test')
    plt.xticks(x, labels)
    plt.ylabel('Cantidad de muestras')
    plt.title('Distribución de Clases (Antes de SMOTE)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Alto', 'Alto'])
    plt.yticks(tick_marks, ['No Alto', 'Alto'])
    thresh = matrix.max() / 2.0
    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, format(matrix[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if matrix[i, j] > thresh else 'black')
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curve(y_true, y_proba, title):
    """Gráfico dual clásico restaurado (PR Curve + ROC Curve)"""
    ap = average_precision_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR AUC = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall — {title}')
    plt.legend()

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
    threshold = np.quantile(y_train, 0.75)
    y_train_c = np.where(y_train > threshold, 1, 0)
    
    # Mostrar el gráfico clásico de distribución antes de balancear
    plot_class_distribution(y_train_c, y_test_c)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train_c)

    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    print('\n' + '=' * 55)
    print('CLASIFICACIÓN — EVALUACIÓN INDIVIDUAL')
    print('=' * 55)

    mejor_auc = 0
    mejor_modelo = ""

    for nombre, modelo in modelos.items():
        modelo.fit(X_train_smote, y_train_smote)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        print(f"\nReporte de Clasificación: {nombre}")
        print(classification_report(y_test_c, y_pred))
        
        roc_auc = roc_auc_score(y_test_c, y_proba)
        if roc_auc > mejor_auc:
            mejor_auc = roc_auc
            mejor_modelo = nombre
            
        plot_confusion_matrix(confusion_matrix(y_test_c, y_pred), f'Matriz Confusión — {nombre}')
        plot_roc_pr_curve(y_test_c, y_proba, nombre)
        
    print(f"\n>> El mejor modelo clásico es: {mejor_modelo} (ROC-AUC: {mejor_auc:.4f})")
    return threshold, X_train_smote, y_train_smote

#DEEP LEARNING — MLP BINARIO
  
def build_deep_learning_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_uniform', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot_training_history_dl(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Entrenamiento MLP — Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

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

    y_pred_dl  = (modelo_dl.predict(X_test_pre) > 0.5).astype(int).flatten()
    y_proba_dl = modelo_dl.predict(X_test_pre).flatten()

    print('\nReporte de clasificación (DL):')
    print(classification_report(y_test_c, y_pred_dl))
    
    plot_confusion_matrix(confusion_matrix(y_test_c, y_pred_dl), 'Matriz de confusión — Deep Learning')
    plot_roc_pr_curve(y_test_c, y_proba_dl, 'Deep Learning')
