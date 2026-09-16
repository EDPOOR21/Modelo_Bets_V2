import pandas as pd
import os
import glob
import numpy as np

# Rutas absolutas para evitar problemas entre Windows y Linux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def determinar_resultado(row):
    """Convierte los goles en formato Quiniela (1, X, 2)"""
    if pd.isna(row['Goles_Local']):
        return np.nan
    if row['Goles_Local'] > row['Goles_Visita']:
        return '1'
    elif row['Goles_Local'] == row['Goles_Visita']:
        return 'X'
    else:
        return '2'

def construir_calendario_maestro():
    print("Iniciando construcción del Calendario Maestro...")
    
    # Buscar los archivos del calendario
    patron = os.path.join(PROCESSED_DIR, "limpio_calendario_*.csv")
    archivos_calendario = glob.glob(patron)
    
    if not archivos_calendario:
        print("[!] No se encontraron calendarios limpios en /data/processed/")
        return
        
    lista_dfs = []
    
    for archivo in archivos_calendario:
        df = pd.read_csv(archivo)
        lista_dfs.append(df)
        
    # Unir todos los calendarios (Stack vertical)
    df_master = pd.concat(lista_dfs, ignore_index=True)
    df_master = df_master.dropna(subset=['Home', 'Away'])
    
    # --- 1. LIMPIEZA DE FORMATOS DE TEXTO ---
    
    # A) Limpiar columna Time para quitar zonas horarias secundarias "13:00 (12:00)" -> "13:00"
    df_master['Time'] = df_master['Time'].astype(str).str.split(' ').str[0]
    
    # B) Reemplazar el guion especial (En Dash) por un guion normal
    df_master['Score'] = df_master['Score'].astype(str).str.replace('–', '-')
    
    # C) Expresión regular para borrar los penales (todo lo que esté entre paréntesis en el Score)
    df_master['Score'] = df_master['Score'].str.replace(r'\(.*?\)', '', regex=True)
    
    # --- 2. EXTRACCIÓN DE GOLES ---
    
    # Filtramos para procesar solo los partidos que ya se jugaron (los que tienen un '-')
    jugados_mask = df_master['Score'].str.contains('-', na=False)
    
    # Inicializar columnas como NaN
    df_master['Goles_Local'] = np.nan
    df_master['Goles_Visita'] = np.nan
    
    # Extraer goles (el float previene errores si hay valores nulos)
    df_master.loc[jugados_mask, 'Goles_Local'] = df_master.loc[jugados_mask, 'Score'].apply(lambda x: float(x.split('-')[0].strip()))
    df_master.loc[jugados_mask, 'Goles_Visita'] = df_master.loc[jugados_mask, 'Score'].apply(lambda x: float(x.split('-')[1].strip()))
    
    # Aplicar la función para generar el Target de nuestro modelo (1, X, 2)
    df_master['Resultado_1X2'] = df_master.apply(determinar_resultado, axis=1)
    
    # --- 3. ORDEN CRONOLÓGICO FINAL ---
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    # Ordenar por fecha y hora real
    df_master = df_master.sort_values(by=['Date', 'Time']).reset_index(drop=True)
    
    # Exportar el CSV final
    ruta_salida = os.path.join(PROCESSED_DIR, "calendario_maestro.csv")
    df_master.to_csv(ruta_salida, index=False)
    
    print(f"\n[v] ÉXITO: Calendario maestro actualizado y parseado correctamente.")
    print(f"Partidos totales en el dataset: {len(df_master)}")
    print(f"Partidos jugados (con resultado): {df_master['Resultado_1X2'].notna().sum()}")
    print(f"Partidos futuros (por predecir): {df_master['Resultado_1X2'].isna().sum()}")

if __name__ == "__main__":
    construir_calendario_maestro()