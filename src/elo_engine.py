import pandas as pd
import os
import math

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Configuración del Sistema Elo
ELO_BASE = 1500
BONO_LOCAL = 50
FACTOR_K = 20  # Velocidad de cambio (20 es estándar para fútbol)

def calcular_probabilidad_esperada(elo_a, elo_b):
    """Calcula la probabilidad de que A le gane a B"""
    return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))

def actualizar_elo(elo_local, elo_visita, resultado):
    """
    Actualiza los puntajes tras el partido.
    resultado: '1' (Gana Local), 'X' (Empate), '2' (Gana Visita)
    """
    # Sumar bono de localía solo para el cálculo matemático del partido
    elo_local_efectivo = elo_local + BONO_LOCAL
    
    # Probabilidades de ganar
    prob_local = calcular_probabilidad_esperada(elo_local_efectivo, elo_visita)
    prob_visita = calcular_probabilidad_esperada(elo_visita, elo_local_efectivo)
    
    # Asignación de puntos reales según el resultado
    if resultado == '1':
        puntos_local, puntos_visita = 1.0, 0.0
    elif resultado == 'X':
        puntos_local, puntos_visita = 0.5, 0.5
    elif resultado == '2':
        puntos_local, puntos_visita = 0.0, 1.0
    else:
        return elo_local, elo_visita # Si no hay resultado, no hay cambio
        
    # Nuevos Elos (El bono de localía no se hereda a la base del equipo)
    nuevo_elo_local = elo_local + FACTOR_K * (puntos_local - prob_local)
    nuevo_elo_visita = elo_visita + FACTOR_K * (puntos_visita - prob_visita)
    
    return round(nuevo_elo_local, 2), round(nuevo_elo_visita, 2)

def procesar_elo_historico():
    print("Iniciando Motor Elo...")
    ruta_master = os.path.join(PROCESSED_DIR, "calendario_maestro.csv")
    
    if not os.path.exists(ruta_master):
        print("[!] Error: No se encontró el calendario maestro.")
        return
        
    df = pd.read_csv(ruta_master)
    
    # Diccionario vivo para rastrear el Elo de cada equipo a lo largo del tiempo
    diccionario_elo = {}
    
    # Nuevas columnas para nuestro dataset
    elos_local_pre = []
    elos_visita_pre = []
    
    for index, row in df.iterrows():
        local = row['Home']
        visita = row['Away']
        resultado = row['Resultado_1X2']
        
        # Inicializar equipos nuevos si no existen en el diccionario
        if local not in diccionario_elo:
            diccionario_elo[local] = ELO_BASE
        if visita not in diccionario_elo:
            diccionario_elo[visita] = ELO_BASE
            
        # 1. GUARDAR EL ELO PREVIO AL PARTIDO (¡Esto es lo que verá el modelo predictivo!)
        elos_local_pre.append(diccionario_elo[local])
        elos_visita_pre.append(diccionario_elo[visita])
        
        # 2. ACTUALIZAR EL ELO (Solo si el partido ya se jugó)
        if pd.notna(resultado):
            nuevo_local, nuevo_visita = actualizar_elo(diccionario_elo[local], diccionario_elo[visita], str(resultado))
            diccionario_elo[local] = nuevo_local
            diccionario_elo[visita] = nuevo_visita

    # Inyectar las variables al DataFrame
    df['Elo_Local_Pre'] = elos_local_pre
    df['Elo_Visita_Pre'] = elos_visita_pre
    
    # Calcular la diferencia (Fuerza Relativa) incluyendo la localía
    df['Diferencia_Elo'] = (df['Elo_Local_Pre'] + BONO_LOCAL) - df['Elo_Visita_Pre']
    
    # Guardar
    ruta_salida = os.path.join(PROCESSED_DIR, "dataset_con_elo.csv")
    df.to_csv(ruta_salida, index=False)
    
    print("[v] ÉXITO: Ranking Elo calculado partido a partido (Sin Data Leakage).")
    print(f"Archivo guardado en: {ruta_salida}")
    
    # Mostrar el Top 5 actual para verificar
    print("\n--- TOP 5 LIGA MX (Ranking Actual) ---")
    top_equipos = sorted(diccionario_elo.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (equipo, puntaje) in enumerate(top_equipos, 1):
        print(f"{i}. {equipo}: {puntaje}")

if __name__ == "__main__":
    procesar_elo_historico()