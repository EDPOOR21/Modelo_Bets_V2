import pandas as pd
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def calcular_features_avanzadas():
    print("Iniciando cálculo de variables avanzadas (Forma, Fatiga y H2H)...")
    ruta_entrada = os.path.join(PROCESSED_DIR, "master_training_data.csv")
    
    if not os.path.exists(ruta_entrada):
        print("[!] Error: No se encontró master_training_data.csv")
        return
        
    df = pd.read_csv(ruta_entrada)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Diccionarios de memoria (El cerebro iterativo)
    estado_equipos = {} # {equipo: {'ultima_fecha': date, 'puntos_ultimos_5': []}}
    historial_h2h = {}  # {'EquipoA-EquipoB': {'EquipoA': [], 'EquipoB': []}}
    
    # Nuevas columnas
    fatiga_local, fatiga_visita = [], []
    forma_local, forma_visita = [], []
    h2h_local_pts = [] 
    
    for index, row in df.iterrows():
        local = row['Home']
        visita = row['Away']
        fecha = row['Date']
        resultado = str(row['Resultado_1X2'])
        
        # 1. Inicializar equipos en memoria si es su primer partido
        for eq in [local, visita]:
            if eq not in estado_equipos:
                estado_equipos[eq] = {'ultima_fecha': fecha - pd.Timedelta(days=14), 'historial_puntos': []}
                
        # Crear llave única para el enfrentamiento directo (ordenada alfabéticamente)
        pair_key = "-".join(sorted([local, visita]))
        if pair_key not in historial_h2h:
            historial_h2h[pair_key] = {local: [], visita: []}
            
        # --- LECTURA PRE-PARTIDO (Lo que verá el modelo predictivo) ---
        
        # A. Fatiga (Topado a 30 días máximo para ignorar pretemporadas)
        dias_l = (fecha - estado_equipos[local]['ultima_fecha']).days
        dias_v = (fecha - estado_equipos[visita]['ultima_fecha']).days
        fatiga_local.append(min(dias_l, 30)) 
        fatiga_visita.append(min(dias_v, 30))
        
        # B. Forma (Suma de puntos en los últimos 5 partidos)
        forma_local.append(sum(estado_equipos[local]['historial_puntos'][-5:]))
        forma_visita.append(sum(estado_equipos[visita]['historial_puntos'][-5:]))
        
        # C. Historial Directo (Puntos sacados por el local en los últimos 5 H2H)
        pts_h2h = sum(historial_h2h[pair_key][local][-5:])
        h2h_local_pts.append(pts_h2h)
        
        # --- ACTUALIZACIÓN POST-PARTIDO (Guardar en memoria para el futuro) ---
        if pd.notna(row['Goles_Local']): 
            estado_equipos[local]['ultima_fecha'] = fecha
            estado_equipos[visita]['ultima_fecha'] = fecha
            
            # Repartir puntos
            pts_l = 3 if resultado == '1' else (1 if resultado == 'X' else 0)
            pts_v = 3 if resultado == '2' else (1 if resultado == 'X' else 0)
            
            estado_equipos[local]['historial_puntos'].append(pts_l)
            estado_equipos[visita]['historial_puntos'].append(pts_v)
            
            historial_h2h[pair_key][local].append(pts_l)
            historial_h2h[pair_key][visita].append(pts_v)

    # Inyectar al DataFrame
    df['Fatiga_Local'] = fatiga_local
    df['Fatiga_Visita'] = fatiga_visita
    df['Forma_Local'] = forma_local
    df['Forma_Visita'] = forma_visita
    df['H2H_Pts_Local'] = h2h_local_pts
    
    # Guardar en un dataset DEFINITIVO para Machine Learning
    ruta_salida = os.path.join(PROCESSED_DIR, "dataset_final_ml.csv")
    df.to_csv(ruta_salida, index=False)
    
    print("[v] ÉXITO: Variables avanzadas (Forma, Fatiga, H2H) calculadas.")
    print(f"Dataset definitivo guardado en: {ruta_salida}")
    print("\nMuestra de las nuevas variables:")
    print(df[['Home', 'Away', 'Forma_Local', 'Forma_Visita', 'Fatiga_Local', 'Fatiga_Visita', 'H2H_Pts_Local']].tail())

if __name__ == "__main__":
    calcular_features_avanzadas()