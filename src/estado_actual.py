import pandas as pd
import os

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final_ml.csv')
EXPORT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'estado_actual_equipos.csv')

def calcular_estado_actual(df):
    print("🔄 Escaneando base de datos y calculando métricas actuales...")
    
    # Identificar a los 18 equipos únicos
    equipos = pd.concat([df['Home'], df['Away']]).unique()
    estado_equipos = []
    
    for equipo in equipos:
        # Aislar el historial de este equipo (sin importar si fue Local o Visita)
        df_equipo = df[(df['Home'] == equipo) | (df['Away'] == equipo)].copy()
        
        if df_equipo.empty:
            continue
            
        # Extraer el último partido para conocer su Elo más reciente
        ultimo_partido = df_equipo.iloc[-1]
        
        # Identificar qué columna leer dependiendo de dónde jugó
        if ultimo_partido['Home'] == equipo:
            elo_actual = ultimo_partido['Elo_Local_Pre']
        else:
            elo_actual = ultimo_partido['Elo_Visita_Pre']
            
        # Calcular el promedio de Tiros a Puerta (SoT) de sus últimos 5 partidos (Termómetro de Forma)
        ultimos_5 = df_equipo.tail(5)
        sot_reciente = ultimos_5.apply(
            lambda row: row['SoT_tiros_Local'] if row['Home'] == equipo else row['SoT_tiros_Visita'], 
            axis=1
        ).mean()
        
        # Empaquetar los datos del equipo
        estado_equipos.append({
            'Equipo': equipo,
            'Elo_Actual': round(elo_actual, 2),
            'Promedio_SoT_5M': round(sot_reciente, 2)
        })
        
    # Crear un DataFrame consolidado y ordenarlo por el mejor Elo
    df_estado = pd.DataFrame(estado_equipos).sort_values(by='Elo_Actual', ascending=False).reset_index(drop=True)
    return df_estado

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df_historico = pd.read_csv(DATA_PATH)
        df_radiografia = calcular_estado_actual(df_historico)
        
        print("\n📊 RADIOGRAFÍA ACTUAL DE LA LIGA MX (TOP 5)")
        print(df_radiografia.head(5).to_string(index=False))
        
        # Guardar en caché para Streamlit
        df_radiografia.to_csv(EXPORT_PATH, index=False)
        print(f"\n✅ Estado actualizado y guardado en: {EXPORT_PATH}")
    else:
        print(f"[!] ERROR: No se encontró el dataset histórico en {DATA_PATH}")