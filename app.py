import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import math  # <-- Nueva librería para calcular Poisson

# 1. Configuración de la página
st.set_page_config(
    page_title="Liga MX Predictor V2.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Funciones de Carga
@st.cache_data
def cargar_estado_equipos():
    ruta = os.path.join("data", "processed", "estado_actual_equipos.csv")
    if os.path.exists(ruta): return pd.read_csv(ruta)
    return None

@st.cache_data
def cargar_historico():
    ruta = os.path.join("data", "processed", "dataset_final_ml.csv")
    if os.path.exists(ruta): return pd.read_csv(ruta)
    return None

@st.cache_resource
def cargar_modelos():
    try:
        base = "models"
        return {
            'features': joblib.load(os.path.join(base, 'features_entrenamiento.pkl')),
            'clf_1x2': joblib.load(os.path.join(base, 'stacking_1x2.pkl')),
            'reg_goles_L': joblib.load(os.path.join(base, 'xg_regressor_local.pkl')),
            'reg_goles_V': joblib.load(os.path.join(base, 'xg_regressor_visita.pkl')),
            'reg_sot_L': joblib.load(os.path.join(base, 'xg_regressor_sot_local.pkl')),
            'reg_sot_V': joblib.load(os.path.join(base, 'xg_regressor_sot_visita.pkl'))
        }
    except Exception as e:
        return None

def main():
    df_estado = cargar_estado_equipos()
    df_hist = cargar_historico()
    modelos = cargar_modelos()
    
    # --- MENÚ LATERAL ---
    st.sidebar.title("💻 SISTEMA CENTRAL v2.0")
    st.sidebar.markdown("---")
    menu = ["📡 Radar Multimercado (Top 10)", "⚖️ Escáner de Cuotas (+EV)", "🧪 Laboratorio H2H", "⏪ Backtesting (Auditoría)"]
    eleccion = st.sidebar.radio("Selecciona Módulo:", menu)
    st.sidebar.markdown("---")
    st.sidebar.info("Motor: Stacking Ensemble | DB: FBref")

    if df_estado is None:
        st.error("🚨 Falta estado_actual_equipos.csv. Ejecuta 'src/estado_actual.py'.")
        return

    lista_equipos = sorted(df_estado['Equipo'].tolist())

    # ==========================================
    # MÓDULO 3: LABORATORIO H2H (CON POISSON)
    # ==========================================
    if eleccion == "🧪 Laboratorio H2H":
        st.title("🧪 Simulador de Partidos (Laboratorio)")
        st.markdown("> Enfrentamiento teórico cara a cara. Motor Stacking Activado.")
        
        col1, col2 = st.columns(2)
        with col1:
            idx_local = lista_equipos.index("América") if "América" in lista_equipos else 0
            equipo_local = st.selectbox("🏠 Selecciona Equipo Local:", lista_equipos, index=idx_local)
        with col2:
            idx_visita = lista_equipos.index("Cruz Azul") if "Cruz Azul" in lista_equipos else 1
            equipo_visita = st.selectbox("✈️ Selecciona Equipo Visitante:", lista_equipos, index=idx_visita)
            
        st.markdown("---")
        
        if st.button("🚀 Ejecutar Simulación", use_container_width=True):
            if equipo_local == equipo_visita:
                st.warning("⚠️ Selecciona dos equipos distintos.")
            elif modelos is None or df_hist is None:
                st.error("🚨 Error cargando los modelos .pkl o el dataset histórico.")
            else:
                stats_local = df_estado[df_estado['Equipo'] == equipo_local].iloc[0]
                stats_visita = df_estado[df_estado['Equipo'] == equipo_visita].iloc[0]
                
                # 1. RECONSTRUCCIÓN DEL VECTOR
                ultimo_local = df_hist[df_hist['Home'] == equipo_local].iloc[-1]
                ultimo_visita = df_hist[df_hist['Away'] == equipo_visita].iloc[-1]
                
                vector = {}
                for col in modelos['features']:
                    if col == 'Diferencia_Elo':
                        vector[col] = stats_local['Elo_Actual'] - stats_visita['Elo_Actual']
                    elif col == 'Elo_Local_Pre':
                        vector[col] = stats_local['Elo_Actual']
                    elif col == 'Elo_Visita_Pre':
                        vector[col] = stats_visita['Elo_Actual']
                    elif '_Local' in col:
                        vector[col] = ultimo_local[col] if col in ultimo_local else 0
                    elif '_Visita' in col:
                        vector[col] = ultimo_visita[col] if col in ultimo_visita else 0
                    else:
                        vector[col] = 0
                        
                X_pred = pd.DataFrame([vector])[modelos['features']]
                
                # 2. INYECCIÓN A LOS MODELOS
                proba_1x2 = modelos['clf_1x2'].predict_proba(X_pred)[0] 
                xg_L = max(0.01, modelos['reg_goles_L'].predict(X_pred)[0]) # Evitar xG negativos
                xg_V = max(0.01, modelos['reg_goles_V'].predict(X_pred)[0])
                
                X_pred_sot = X_pred.drop(columns=['SoT_tiros_Local', 'SoT_tiros_Visita'])
                sot_L = modelos['reg_sot_L'].predict(X_pred_sot)[0]
                sot_V = modelos['reg_sot_V'].predict(X_pred_sot)[0]
                
                # --- MATEMÁTICAS APLICADAS ---
                # A) Corrección de Tiros a Puerta (Promedio por partido)
                sot_L_partido = sot_L / 5.0
                sot_V_partido = sot_V / 5.0
                
                # B) Marcador Sugerido (Redondeo de xG)
                marcador_L = int(round(xg_L))
                marcador_V = int(round(xg_V))
                
                # C) Distribución de Poisson para Over/Under 2.5
                total_xg = xg_L + xg_V
                prob_0 = math.exp(-total_xg)
                prob_1 = total_xg * math.exp(-total_xg)
                prob_2 = ((total_xg**2) * math.exp(-total_xg)) / 2
                prob_under_25 = (prob_0 + prob_1 + prob_2) * 100
                prob_over_25 = 100 - prob_under_25
                
                # 3. RENDERIZADO HACKER
                st.success("✅ Predicción generada con éxito.")
                st.markdown("### 🤖 Radiografía del Algoritmo (1X2)")
                
                r1, r2, r3 = st.columns(3)
                r1.metric("Prob. Victoria Local", f"{proba_1x2[2]*100:.1f}%") 
                r2.metric("Prob. Empate", f"{proba_1x2[1]*100:.1f}%")
                r3.metric("Prob. Victoria Visita", f"{proba_1x2[0]*100:.1f}%")
                
                st.markdown("---")
                
                # Nueva sección de Goles y Marcador
                st.markdown("### 🥅 Análisis de Goles y Marcador")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"xG {equipo_local}", f"{xg_L:.2f}")
                m2.metric(f"xG {equipo_visita}", f"{xg_V:.2f}")
                m3.metric("Marcador Exacto Sugerido", f"{marcador_L} - {marcador_V}")
                m4.metric("Mercado Over 2.5", f"{prob_over_25:.1f}%", f"{prob_under_25:.1f}% Under", delta_color="off")
                
                st.markdown("---")
                
                # Sección de Volumen Ofensivo Corregida
                st.markdown("### 🎯 Proyecciones de Volumen (90 Min)")
                g1, g2 = st.columns(2)
                g1.metric(f"Tiros a Puerta ({equipo_local})", f"{sot_L_partido:.1f}")
                g2.metric(f"Tiros a Puerta ({equipo_visita})", f"{sot_V_partido:.1f}")

    # ==========================================
    # MÓDULO 1: RADAR MULTIMERCADO (TOP 10)
    # ==========================================
    elif eleccion == "📡 Radar Multimercado (Top 10)":
        st.title("📡 Radar Semanal: Armado de Jornada")
        st.markdown("> Configura los 9 enfrentamientos de la jornada. El motor evaluará todos los cruces en lote.")
        
        # Formulario para armar la jornada
        with st.form("form_jornada"):
            st.subheader("🗓️ Selectores de Partidos")
            partidos_seleccionados = []
            
            # Generar 9 filas para los 9 partidos de la Liga MX
            for i in range(9):
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.markdown(f"**Partido {i+1}**")
                with c2:
                    # Asignar índices por defecto diferentes para evitar cruces idénticos al inicio
                    idx_L = (i * 2) % len(lista_equipos)
                    local = st.selectbox(f"Local {i+1}", lista_equipos, index=idx_L, key=f"L_{i}", label_visibility="collapsed")
                with c3:
                    idx_V = (i * 2 + 1) % len(lista_equipos)
                    visita = st.selectbox(f"Visita {i+1}", lista_equipos, index=idx_V, key=f"V_{i}", label_visibility="collapsed")
                
                partidos_seleccionados.append((local, visita))
                
            st.markdown("---")
            ejecutar_radar = st.form_submit_button("⚡ Escanear Jornada Completa", use_container_width=True)

        # Lógica de procesamiento en lote
        if ejecutar_radar:
            if modelos is None or df_hist is None:
                st.error("🚨 Error cargando los modelos .pkl o el dataset histórico.")
            else:
                resultados_jornada = []
                barra_progreso = st.progress(0)
                
                for i, (equipo_local, equipo_visita) in enumerate(partidos_seleccionados):
                    if equipo_local == equipo_visita:
                        continue # Saltar partidos inválidos
                        
                    stats_local = df_estado[df_estado['Equipo'] == equipo_local].iloc[0]
                    stats_visita = df_estado[df_estado['Equipo'] == equipo_visita].iloc[0]
                    
                    ultimo_local = df_hist[df_hist['Home'] == equipo_local].iloc[-1]
                    ultimo_visita = df_hist[df_hist['Away'] == equipo_visita].iloc[-1]
                    
                    # 1. RECONSTRUCCIÓN DEL VECTOR
                    vector = {}
                    for col in modelos['features']:
                        if col == 'Diferencia_Elo':
                            vector[col] = stats_local['Elo_Actual'] - stats_visita['Elo_Actual']
                        elif col == 'Elo_Local_Pre':
                            vector[col] = stats_local['Elo_Actual']
                        elif col == 'Elo_Visita_Pre':
                            vector[col] = stats_visita['Elo_Actual']
                        elif '_Local' in col:
                            vector[col] = ultimo_local[col] if col in ultimo_local else 0
                        elif '_Visita' in col:
                            vector[col] = ultimo_visita[col] if col in ultimo_visita else 0
                        else:
                            vector[col] = 0
                            
                    X_pred = pd.DataFrame([vector])[modelos['features']]
                    
                    # 2. INYECCIÓN
                    proba_1x2 = modelos['clf_1x2'].predict_proba(X_pred)[0] 
                    xg_L = max(0.01, modelos['reg_goles_L'].predict(X_pred)[0])
                    xg_V = max(0.01, modelos['reg_goles_V'].predict(X_pred)[0])
                    
                    X_pred_sot = X_pred.drop(columns=['SoT_tiros_Local', 'SoT_tiros_Visita'])
                    sot_L = modelos['reg_sot_L'].predict(X_pred_sot)[0] / 5.0
                    sot_V = modelos['reg_sot_V'].predict(X_pred_sot)[0] / 5.0
                    
                    # Poisson Over 2.5
                    total_xg = xg_L + xg_V
                    prob_under_25 = (math.exp(-total_xg) + total_xg * math.exp(-total_xg) + ((total_xg**2) * math.exp(-total_xg)) / 2) * 100
                    
                    # 3. ALMACENAMIENTO (Ahora como valores numéricos puros)
                    resultados_jornada.append({
                        "Partido": f"{equipo_local} vs {equipo_visita}",
                        "1 (Local)": round(proba_1x2[2]*100, 1),
                        "X (Empate)": round(proba_1x2[1]*100, 1),
                        "2 (Visita)": round(proba_1x2[0]*100, 1),
                        "Marcador Sugerido": f"{int(round(xg_L))} - {int(round(xg_V))}",
                        "xG Total": round(total_xg, 2),
                        "Over 2.5": round(100 - prob_under_25, 1),
                        "SoT Local": round(sot_L, 1),
                        "SoT Visita": round(sot_V, 1)
                    })
                    
                    barra_progreso.progress((i + 1) / 9)
                
                # 4. RENDERIZADO DEL RADAR (Con formato condicional)
                st.success("✅ Análisis de la jornada completado.")
                df_resultados = pd.DataFrame(resultados_jornada)
                
                # Crear el mapa de calor (Styler) con formato numérico estricto
                tabla_estilizada = df_resultados.style.background_gradient(
                    cmap='Greens', 
                    subset=['1 (Local)', 'X (Empate)', '2 (Visita)'], 
                    axis=1  
                ).background_gradient(
                    cmap='Blues', 
                    subset=['Over 2.5'] 
                ).format({
                    '1 (Local)': "{:.1f}%",
                    'X (Empate)': "{:.1f}%",
                    '2 (Visita)': "{:.1f}%",
                    'Over 2.5': "{:.1f}%",
                    'xG Total': "{:.2f}",   # <-- Nuevo
                    'SoT Local': "{:.1f}",  # <-- Nuevo
                    'SoT Visita': "{:.1f}"  # <-- Nuevo
                })
                
                st.markdown("### 📊 Tabla de Predicciones")
                st.dataframe(tabla_estilizada, use_container_width=True, hide_index=True)

    elif eleccion == "⚖️ Escáner de Cuotas (+EV)":
        st.title("⚖️ Buscador de Valor (+EV)")
        st.write("*(En construcción...)*")
    elif eleccion == "⏪ Backtesting (Auditoría)":
        st.title("⏪ Auditoría del Modelo")
        st.write("*(En construcción...)*")

if __name__ == "__main__":
    main()