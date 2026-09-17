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

def entrenar_modelos_volumen():
    print("\n===========================================================")
    print("🌪️ INICIANDO ENTRENAMIENTO DE VOLUMEN OFENSIVO (SoT) 🌪️")
    print("===========================================================")
    
    df = pd.read_csv(DATA_PATH)
    
    # Nombres EXACTOS extraídos de tu consola
    targets = ['SoT_tiros_Local', 'SoT_tiros_Visita']
    
    # Validar cuáles de estas columnas realmente existen en tu dataset actual
    targets_disponibles = [t for t in targets if t in df.columns]
    
    if not targets_disponibles:
        print("[!] No se encontraron las columnas. Revisa los nombres.")
        return

    df_train = df.dropna(subset=targets_disponibles).copy()
    
    # 1. Variables Predictivas (Features)
    features_base = [
        'Elo_Local_Pre', 'Elo_Visita_Pre', 'Diferencia_Elo'
    ]
    
    # Excluimos cualquier variable del "futuro" o que sea texto
    columnas_prohibidas = targets + ['Goles_Local', 'Goles_Visita', 'Resultado_1X2', 'Score', 'Match Report', 'Notes', 'Referee', 'Venue']
    
    features_tacticas = [col for col in df_train.columns if ('_Local' in col or '_Visita' in col) 
                         and col not in features_base 
                         and col not in columnas_prohibidas]
                         
    columnas_validas = [col for col in (features_base + features_tacticas) if col in df_train.columns]
    X = df_train[columnas_validas].select_dtypes(include=['number'])
    
    print(f"Entrenando con {len(X)} partidos y {X.shape[1]} variables...")

    # 2. Entrenar y guardar un modelo independiente para cada métrica
    for target in targets_disponibles:
        y = df_train[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nEntrenando Regresor para: {target}...")
        modelo = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        modelo.fit(X_train, y_train)
        
        pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        
        print(f"📊 Margen de Error ({target}): +/- {mae:.2f} tiros a puerta")
        
        # Limpiar el nombre para guardar el archivo (ej. de 'SoT_tiros_Local' a 'sot_local')
        nombre_limpio = target.replace('_tiros', '').lower()
        joblib.dump(modelo, os.path.join(MODELS_DIR, f'xg_regressor_{nombre_limpio}.pkl'))

    print("\n✅ ENTRENAMIENTO FINALIZADO")
    print("💾 Modelos guardados en /models/")
    print("===========================================================\n")

if __name__ == "__main__":
    entrenar_modelos_volumen()