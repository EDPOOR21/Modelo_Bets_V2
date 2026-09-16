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

def entrenar_modelo_1x2():
    print("Cargando dataset definitivo y preparando el entorno de ML...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Limpieza final: Quitar partidos futuros (los que aún no tienen resultado)
    df_train = df.dropna(subset=['Resultado_1X2']).copy()
    
    # 2. Mapear el target para XGBoost (Requiere valores numéricos: 0, 1, 2)
    # 0 = Gana Local ('1'), 1 = Empate ('X'), 2 = Gana Visita ('2')
    mapa_resultados = {'1': 0, 'X': 1, '2': 2}
    df_train['Target'] = df_train['Resultado_1X2'].map(mapa_resultados)
    
    # 3. Seleccionar las variables predictivas (Features Base + Avanzadas)
    features = [
        'Elo_Local_Pre', 'Elo_Visita_Pre', 'Diferencia_Elo', 
        'Poisson_xG_Local', 'Poisson_xG_Visita', 
        'Forma_Local', 'Forma_Visita', 
        'Fatiga_Local', 'Fatiga_Visita', 'H2H_Pts_Local'
    ]
    
    X = df_train[features]
    y = df_train['Target']
    
    # 4. Dividir en Entrenamiento (80%) y Validación (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Partidos para entrenar: {len(X_train)} | Partidos para validar: {len(X_test)}")
    print("\nInicializando Nivel 1 (XGBoost + Random Forest) y Nivel 2 (Regresión Logística)...")
    
    # 5. Definir los Modelos Base (Nivel 1)
    estimadores_base = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # 6. Construir el Stacking Ensemble
    modelo_stacking = StackingClassifier(
        estimators=estimadores_base,
        final_estimator=LogisticRegression(),
        cv=5, # Validación cruzada interna para evitar sobreajuste
        passthrough=False # El Nivel 2 solo verá las predicciones del Nivel 1
    )
    
    # 7. ENTRENAMIENTO (Aquí ocurre la magia)
    print("Entrenando el ensamble... (Esto puede tardar unos segundos)")
    modelo_stacking.fit(X_train, y_train)
    
    # 8. Evaluación
    y_pred = modelo_stacking.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"\n[v] ÉXITO: Entrenamiento finalizado.")
    print(f"Precisión global del Ensamble en datos no vistos: {precision:.2%}")
    print("\nReporte de Clasificación (0: Local, 1: Empate, 2: Visita):")
    print(classification_report(y_test, y_pred))
    
    # 9. Guardar el modelo en disco (.pkl)
    ruta_modelo = os.path.join(MODELS_DIR, 'stacking_1x2.pkl')
    joblib.dump(modelo_stacking, ruta_modelo)
    print(f"Modelo compilado y guardado de forma segura en: {ruta_modelo}")

if __name__ == "__main__":
    entrenar_modelo_1x2()