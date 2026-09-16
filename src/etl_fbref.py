import pandas as pd
import os
import glob

# 1. Definir rutas absolutas para evitar errores en Windows
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def procesar_csvs_crudos():
    # Buscar todos los archivos CSV en la carpeta raw
    archivos_csv = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    
    if not archivos_csv:
        print("[!] No se encontraron archivos CSV en la carpeta /data/raw/")
        return

    print(f"Iniciando limpieza de {len(archivos_csv)} archivos...\n")

    for ruta_archivo in archivos_csv:
        nombre_archivo = os.path.basename(ruta_archivo)
        
        try:
            # Intentamos leer el CSV asumiendo estructura simple
            df = pd.read_csv(ruta_archivo)
            
            # Validar doble cabecera: Si 'Squad' o 'Wk' (Jornada) no están en las columnas,
            # significa que FBref puso una cabecera inútil en la fila 0.
            if 'Squad' not in df.columns and 'Wk' not in df.columns:
                df = pd.read_csv(ruta_archivo, header=1)
            
            # LIMPIEZA CLAVE: Eliminar columnas que estén 100% vacías (NaN)
            # Esto borrará todas esas columnas de "Touches", "Take-Ons" que viste vacías.
            df = df.dropna(axis=1, how='all')
            
            # Limpiar espacios en los nombres de las columnas
            df.columns = df.columns.str.strip()

            # Guardar el archivo limpio en /data/processed/
            ruta_salida = os.path.join(PROCESSED_DIR, f"limpio_{nombre_archivo}")
            df.to_csv(ruta_salida, index=False)
            
            print(f"[v] Limpiado: {nombre_archivo} | Columnas retenidas: {len(df.columns)}")
            
        except Exception as e:
            print(f"[x] Error al procesar {nombre_archivo}: {e}")

if __name__ == "__main__":
    procesar_csvs_crudos()
    print("\n¡Proceso ETL completado! Revisa tu carpeta /data/processed/")