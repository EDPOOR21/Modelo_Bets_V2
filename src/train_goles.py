import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final_ml.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def entrenar_modelos_goles():
    print("\n===========================================================")
    print("🎯 INICIANDO ENTRENAMIENTO DE REGRESIÓN DE GOLES (xG) 🎯")
    print("===========================================================")
    
    df = pd.read_csv(DATA_PATH)
    df_train = df.dropna(subset=['Goles_Local', 'Goles_Visita']).copy()
    
    # 1. Seleccionar Variables Predictivas (Features)
    features_base = [
        'Elo_Local_Pre', 'Elo_Visita_Pre', 'Diferencia_Elo', 
        'Poisson_xG_Local', 'Poisson_xG_Visita', 
        'Forma_Local', 'Forma_Visita', 
        'Fatiga_Local', 'Fatiga_Visita'
    ]
    
    features_tacticas = [col for col in df_train.columns if ('_Local' in col or '_Visita' in col) 
                         and col not in features_base 
                         and col not in ['Goles_Local', 'Goles_Visita', 'Resultado_1X2']]
                         
    columnas_validas = [col for col in (features_base + features_tacticas) if col in df_train.columns]
    X = df_train[columnas_validas].select_dtypes(include=['number'])
    
    # Variables Objetivo (Targets)
    y_local = df_train['Goles_Local']
    y_visita = df_train['Goles_Visita']
    
    print(f"Entrenando con {len(X)} partidos y {X.shape[1]} variables...")

    # 2. División de datos (Entrenamiento y Validación)
    X_train, X_test, y_local_train, y_local_test = train_test_split(X, y_local, test_size=0.2, random_state=42)
    _, _, y_visita_train, y_visita_test = train_test_split(X, y_visita, test_size=0.2, random_state=42)

    # 3. Inicializar y Entrenar XGBRegressor
    print("\nEntrenando Regresor Local...")
    modelo_goles_local = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    modelo_goles_local.fit(X_train, y_local_train)
    
    print("Entrenando Regresor Visitante...")
    modelo_goles_visita = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    modelo_goles_visita.fit(X_train, y_visita_train)

    # 4. Evaluación de Precisión (Margen de Error)
    pred_local = modelo_goles_local.predict(X_test)
    pred_visita = modelo_goles_visita.predict(X_test)
    
    mae_local = mean_absolute_error(y_local_test, pred_local)
    mae_visita = mean_absolute_error(y_visita_test, pred_visita)

    print("\n✅ ENTRENAMIENTO FINALIZADO")
    print("📊 Margen de Error Promedio (Goles Reales vs Predicción):")
    print(f"🏠 Local   : +/- {mae_local:.2f} goles por partido")
    print(f"✈️ Visitante: +/- {mae_visita:.2f} goles por partido")

    # 5. Guardar los modelos
    joblib.dump(modelo_goles_local, os.path.join(MODELS_DIR, 'xg_regressor_local.pkl'))
    joblib.dump(modelo_goles_visita, os.path.join(MODELS_DIR, 'xg_regressor_visita.pkl'))
    
    print("\n💾 Modelos guardados en /models/")
    print("===========================================================\n")

if __name__ == "__main__":
    entrenar_modelos_goles()