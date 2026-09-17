import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final_ml.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Asegurar que exista la carpeta models/
os.makedirs(MODELS_DIR, exist_ok=True)

def entrenar_modelo_1x2():
    print("\n===========================================================")
    print("🧠 INICIANDO ENTRENAMIENTO DEL STACKING ENSEMBLE (1X2) 🧠")
    print("===========================================================")
    print("Cargando dataset definitivo y preparando el entorno de ML...")
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. Limpieza final: Quitar partidos futuros
    df_train = df.dropna(subset=['Resultado_1X2']).copy()
    
    # 2. Mapear el target para XGBoost (0 = Gana Local, 1 = Empate, 2 = Gana Visita)
    mapa_resultados = {'1': 0, 'X': 1, '2': 2}
    df_train['Target'] = df_train['Resultado_1X2'].map(mapa_resultados)
    
    # 3. Seleccionar las variables predictivas (Features Base + Avanzadas)
    features_base = [
        'Elo_Local_Pre', 'Elo_Visita_Pre', 'Diferencia_Elo', 
        'Poisson_xG_Local', 'Poisson_xG_Visita', 
        'Forma_Local', 'Forma_Visita', 
        'Fatiga_Local', 'Fatiga_Visita', 'H2H_Pts_Local'
    ]
    
    # Recolectar dinámicamente todo el ADN táctico cruzado en el paso ETL
    features_tacticas = [col for col in df_train.columns if ('_Local' in col or '_Visita' in col) 
                         and col not in features_base 
                         and col not in ['Goles_Local', 'Goles_Visita']]
    
    features_totales = features_base + features_tacticas
    
    # Filtrar solo columnas numéricas y que existan en el DataFrame actual
    columnas_validas = [col for col in features_totales if col in df_train.columns]
    X = df_train[columnas_validas].select_dtypes(include=['number'])
    y = df_train['Target']
    
    # 4. Dividir en Entrenamiento (80%) y Validación (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Partidos para entrenar: {len(X_train)} | Partidos para validar: {len(X_test)}")
    print(f"Total de variables predictivas activas: {X_train.shape[1]}")
    print("\nInicializando Nivel 1 (XGBoost + Random Forest) y Nivel 2 (Regresión Logística)...")
    
    # 5. Definir los Modelos Base (Nivel 1) con pesos balanceados
    estimadores_base = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42, class_weight='balanced'))
    ]
    
    # 6. Construir el Stacking Ensemble con regresión balanceada
    modelo_stacking = StackingClassifier(
        estimators=estimadores_base,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5,
        passthrough=False
    )
    
    # 7. ENTRENAMIENTO
    print("Entrenando el ensamble... (Esto puede tardar unos segundos)")
    modelo_stacking.fit(X_train, y_train)
    
    # 8. Evaluación
    y_pred = modelo_stacking.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ ÉXITO: Entrenamiento finalizado.")
    print(f"🎯 Precisión global del Ensamble en datos no vistos: {precision:.2%}")
    print("\nReporte de Clasificación (0: Local, 1: Empate, 2: Visita):")
    # Añadido zero_division=0 para suprimir advertencias rojas
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 9. Guardar el modelo y las features en disco
    ruta_modelo = os.path.join(MODELS_DIR, 'stacking_1x2.pkl')
    ruta_features = os.path.join(MODELS_DIR, 'features_entrenamiento.pkl')
    
    joblib.dump(modelo_stacking, ruta_modelo)
    joblib.dump(list(X.columns), ruta_features)
    
    print(f"💾 Modelo compilado y guardado en: {ruta_modelo}")
    print("===========================================================\n")

if __name__ == "__main__":
    entrenar_modelo_1x2()