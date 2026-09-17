import pandas as pd
import numpy as np
import os
import joblib

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final_ml.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stacking_1x2.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'features_entrenamiento.pkl')

def obtener_ultimos_datos(equipo_local, equipo_visita, df):
    """
    Busca la versión más reciente (último partido) de cada equipo en el dataset 
    para extraer su Elo, Poisson y ADN Táctico actual.
    """
    # Aislar datos más recientes de cada equipo
    stats_local = df[(df['Home'] == equipo_local) | (df['Away'] == equipo_local)].tail(1)
    stats_visita = df[(df['Home'] == equipo_visita) | (df['Away'] == equipo_visita)].tail(1)
    
    if stats_local.empty or stats_visita.empty:
        return None
        
    # Construir un diccionario temporal con la mezcla de ambos
    datos_partido = {}
    
    # Extraer variables base 
    elo_loc = stats_local['Elo_Local_Pre'].values[0] if stats_local['Home'].values[0] == equipo_local else stats_local['Elo_Visita_Pre'].values[0]
    elo_vis = stats_visita['Elo_Local_Pre'].values[0] if stats_visita['Home'].values[0] == equipo_visita else stats_visita['Elo_Visita_Pre'].values[0]
    
    datos_partido['Elo_Local_Pre'] = elo_loc
    datos_partido['Elo_Visita_Pre'] = elo_vis
    datos_partido['Diferencia_Elo'] = elo_loc - elo_vis
    
    return pd.DataFrame([datos_partido])

def predecir_enfrentamiento(equipo_local, equipo_visita):
    print(f"\n⚽ PREPARANDO PREDICCIÓN: {equipo_local} (L) vs {equipo_visita} (V) ⚽")
    
    # 1. Cargar el modelo y las columnas exactas que exige
    try:
        modelo = joblib.load(MODEL_PATH)
        columnas_requeridas = joblib.load(FEATURES_PATH)
        df_historico = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"[!] Error al cargar archivos: {e}")
        return

    # 2. Extraer stats recientes
    df_pred = obtener_ultimos_datos(equipo_local, equipo_visita, df_historico)
    if df_pred is None:
        print(f"[!] No se encontraron datos históricos suficientes para estos equipos.")
        return

    # 3. Formatear las columnas exactamente como las pide el modelo
    for col in columnas_requeridas:
        if col not in df_pred.columns:
            df_pred[col] = df_historico[col].mean()

    # Ordenar estrictamente
    X_pred = df_pred[columnas_requeridas]

    # 4. Generar Probabilidades
    probabilidades = modelo.predict_proba(X_pred)[0]
    
    print("\n==========================================")
    print("📊 RADIOGRAFÍA DE PROBABILIDADES (1X2)")
    print("==========================================")
    print(f"🏠 Victoria {equipo_local} (1) : {probabilidades[0]:.2%}")
    print(f"⚖️ Empate (X)             : {probabilidades[1]:.2%}")
    print(f"✈️ Victoria {equipo_visita} (2) : {probabilidades[2]:.2%}")
    print("==========================================\n")

if __name__ == "__main__":
    # Nombres exactos extraídos de la base de datos
    predecir_enfrentamiento("América", "UNAM")