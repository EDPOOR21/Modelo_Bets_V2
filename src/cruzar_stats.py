import pandas as pd
import os

# --- RUTAS BASE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def cargar_y_unir_categoria(categoria):
    """
    Busca los archivos limpios de una categoría e inyecta una columna 
    identificadora de la temporada de la que provienen.
    """
    temporadas = ['24_25', '25_26', 'actual']
    dfs = []
    
    for temp in temporadas:
        ruta = os.path.join(PROCESSED_DIR, f'limpio_{categoria}_{temp}.csv')
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            # Creamos la llave maestra para cruzar
            df['Temporada_Join'] = temp  
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def asignar_temporada(fecha):
    """
    Mapea la fecha exacta del partido a la temporada futbolística de la Liga MX.
    """
    año = fecha.year
    mes = fecha.month
    
    if (año == 2024 and mes >= 6) or (año == 2025 and mes <= 5):
        return '24_25'
    elif (año == 2025 and mes >= 6) or (año == 2026 and mes <= 5):
        return '25_26'
    elif año == 2026 and mes >= 6:
        return 'actual'
    return 'actual' # Fallback de seguridad

def ejecutar_cruce_maestro():
    print("\n===========================================================")
    print("🧬 INICIANDO CRUCE DE ADN TÁCTICO (PROMEDIOS DE TEMPORADA) 🧬")
    print("===========================================================")

    # 1. Cargar la base
    ruta_base = os.path.join(PROCESSED_DIR, 'dataset_con_elo.csv')
    df_base = pd.read_csv(ruta_base)
    df_base['Date'] = pd.to_datetime(df_base['Date'])
    
    # Asignar la llave de temporada a cada partido basado en su fecha
    df_base['Temporada_Join'] = df_base['Date'].apply(asignar_temporada)
    
    # 2. Las categorías que vamos a inyectar
    categorias = ['tiros', 'pases', 'posesion', 'misc']
    
    for cat in categorias:
        df_stats = cargar_y_unir_categoria(cat)
        
        if df_stats is not None:
            print(f"🔄 Cruzando estadísticas de: {cat.upper()}...")
            
            # La llave ahora es la Temporada y el Equipo
            cols_llave = ['Temporada_Join', 'Squad']
            cols_metricas = [c for c in df_stats.columns if c not in cols_llave and c != 'Equipo']
            
            # --- PREPARAR DATOS DEL LOCAL ---
            df_stats_local = df_stats[cols_llave + cols_metricas].copy()
            df_stats_local.rename(columns={'Squad': 'Home'}, inplace=True)
            df_stats_local.rename(columns={c: f"{c}_{cat}_Local" for c in cols_metricas}, inplace=True)
            
            # --- PREPARAR DATOS DE LA VISITA ---
            df_stats_visita = df_stats[cols_llave + cols_metricas].copy()
            df_stats_visita.rename(columns={'Squad': 'Away'}, inplace=True)
            df_stats_visita.rename(columns={c: f"{c}_{cat}_Visita" for c in cols_metricas}, inplace=True)
            
            # --- EJECUTAR INNER JOIN ---
            df_base = pd.merge(df_base, df_stats_local, on=['Temporada_Join', 'Home'], how='left')
            df_base = pd.merge(df_base, df_stats_visita, on=['Temporada_Join', 'Away'], how='left')
            
    # 3. Limpieza de seguridad adaptativa
    # Llenar nulos numéricos con 0 (para que XGBoost funcione)
    cols_numericas = df_base.select_dtypes(include=['number']).columns
    df_base[cols_numericas] = df_base[cols_numericas].fillna(0)
    
    # Llenar nulos de texto con 'N/A' (para no romper tipos de datos)
    cols_texto = df_base.select_dtypes(include=['object', 'string']).columns
    df_base[cols_texto] = df_base[cols_texto].fillna('N/A')
    
    df_base.drop(columns=['Temporada_Join'], inplace=True) # Borramos la columna auxiliar
            
    # 4. Exportar el verdadero Dataset Final listo para Machine Learning
    ruta_salida = os.path.join(PROCESSED_DIR, 'dataset_final_ml.csv')
    df_base.to_csv(ruta_salida, index=False)
    
    print(f"\n✅ ¡Cruce completado con éxito!")
    print(f"📊 El dataset final tiene ahora {df_base.shape[1]} columnas y {df_base.shape[0]} partidos históricos.")
    print(f"💾 Guardado en: {ruta_salida}")
    print("===========================================================\n")

if __name__ == "__main__":
    ejecutar_cruce_maestro()