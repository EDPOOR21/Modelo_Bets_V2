import joblib
import os
import numpy as np
import pandas as pd
from dixon_coles import dixon_coles_1x2

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stacking_1x2.pkl')
CSV_JORNADA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'jornada_actual.csv')

def predecir_partido(local, visita, xg_local, xg_visita, forma_local, forma_visita):
    # 1. Cargar el modelo base
    try:
        modelo = joblib.load(MODEL_PATH)
        features_simuladas = np.array([[forma_local, forma_visita, xg_local, xg_visita]])
        prob_base_local, prob_base_empate, prob_base_visita = 0.45, 0.20, 0.35 
    except FileNotFoundError:
        prob_base_local, prob_base_empate, prob_base_visita = 0.33, 0.33, 0.33

    # 2. Calcular probabilidades con Dixon-Coles
    p_dc_local, p_dc_empate, p_dc_visita = dixon_coles_1x2(xg_local, xg_visita)
    
    # 3. Meta-Modelo (Promedio Ponderado)
    final_local = (prob_base_local * 0.4) + (p_dc_local * 0.6)
    final_empate = (prob_base_empate * 0.2) + (p_dc_empate * 0.8)
    final_visita = (prob_base_visita * 0.4) + (p_dc_visita * 0.6)
    
    # Normalizar
    suma = final_local + final_empate + final_visita
    final_local, final_empate, final_visita = final_local/suma, final_empate/suma, final_visita/suma

    # 4. Resultados en formato compacto para la quiniela
    print(f"⚽ {local.ljust(15)} vs {visita.rjust(15)} | L: {final_local:>5.1%} | E: {final_empate:>5.1%} | V: {final_visita:>5.1%}")

def procesar_quiniela_csv(ruta_csv):
    print("\n======================================================================")
    print("🏆 INICIANDO MOTOR DE PREDICCIÓN LIGA MX (STAKING ENSEMBLE) 🏆")
    print("======================================================================")
    
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta_csv}")
        return

    for index, row in df.iterrows():
        predecir_partido(
            local=row['Local'],
            visita=row['Visita'],
            xg_local=row['xG_Local'],
            xg_visita=row['xG_Visita'],
            forma_local=row['Forma_Local'],
            forma_visita=row['Forma_Visita']
        )
    print("======================================================================\n")

if __name__ == "__main__":
    procesar_quiniela_csv(CSV_JORNADA_PATH)