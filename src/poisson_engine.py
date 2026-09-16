import pandas as pd
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def calcular_esperanza_goles():
    print("Iniciando Motor de Poisson (Fuerza Ofensiva y Defensiva)...")
    ruta_entrada = os.path.join(PROCESSED_DIR, "dataset_con_elo.csv")
    
    if not os.path.exists(ruta_entrada):
        print("[!] Error: No se encontró dataset_con_elo.csv")
        return
        
    df = pd.read_csv(ruta_entrada)
    
    # Diccionario para rastrear el acumulado de cada equipo
    # Estructura: {'América': {'Anotados': 0, 'Recibidos': 0, 'Partidos': 0}}
    stats_equipos = {}
    
    # Listas para guardar las nuevas variables predictivas
    lambda_local_lista = []
    lambda_visita_lista = []
    
    for index, row in df.iterrows():
        local = row['Home']
        visita = row['Away']
        goles_l = row['Goles_Local']
        goles_v = row['Goles_Visita']
        
        # Inicializar equipos si no existen
        for equipo in [local, visita]:
            if equipo not in stats_equipos:
                stats_equipos[equipo] = {'Anotados': 0, 'Recibidos': 0, 'Partidos': 0}
                
        # 1. CÁLCULO PRE-PARTIDO (Lo que ve el modelo)
        # Promedios del Local
        partidos_l = stats_equipos[local]['Partidos']
        prom_anotados_l = stats_equipos[local]['Anotados'] / partidos_l if partidos_l > 0 else 1.0
        prom_recibidos_l = stats_equipos[local]['Recibidos'] / partidos_l if partidos_l > 0 else 1.0
        
        # Promedios del Visita
        partidos_v = stats_equipos[visita]['Partidos']
        prom_anotados_v = stats_equipos[visita]['Anotados'] / partidos_v if partidos_v > 0 else 1.0
        prom_recibidos_v = stats_equipos[visita]['Recibidos'] / partidos_v if partidos_v > 0 else 1.0
        
        # Fórmula Proxy de Poisson: Fuerza Ofensiva * Debilidad Defensiva Rival
        # Si el local anota mucho y la visita recibe mucho, el xG se dispara.
        xg_esperado_local = prom_anotados_l * prom_recibidos_v
        xg_esperado_visita = prom_anotados_v * prom_recibidos_l
        
        lambda_local_lista.append(round(xg_esperado_local, 3))
        lambda_visita_lista.append(round(xg_esperado_visita, 3))
        
        # 2. ACTUALIZAR ESTADÍSTICAS (Solo si el partido ya se jugó)
        if pd.notna(goles_l) and pd.notna(goles_v):
            # Actualizar Local
            stats_equipos[local]['Anotados'] += goles_l
            stats_equipos[local]['Recibidos'] += goles_v
            stats_equipos[local]['Partidos'] += 1
            
            # Actualizar Visita
            stats_equipos[visita]['Anotados'] += goles_v
            stats_equipos[visita]['Recibidos'] += goles_l
            stats_equipos[visita]['Partidos'] += 1

    # Inyectar las variables al DataFrame
    df['Poisson_xG_Local'] = lambda_local_lista
    df['Poisson_xG_Visita'] = lambda_visita_lista
    
    # Crear variable de "Proyección de Goles Totales" (Para el mercado Over/Under)
    df['Proyeccion_Goles_Totales'] = df['Poisson_xG_Local'] + df['Poisson_xG_Visita']
    
    # Guardar sobre el mismo archivo maestro de entrenamiento
    ruta_salida = os.path.join(PROCESSED_DIR, "master_training_data.csv")
    df.to_csv(ruta_salida, index=False)
    
    print("[v] ÉXITO: Variables de Poisson (xG) agregadas al dataset.")
    print(f"Dataset final de entrenamiento guardado en: {ruta_salida}")
    print("\nMuestra de las variables predictivas creadas:")
    print(df[['Home', 'Away', 'Elo_Local_Pre', 'Elo_Visita_Pre', 'Poisson_xG_Local', 'Poisson_xG_Visita']].tail())

if __name__ == "__main__":
    calcular_esperanza_goles()