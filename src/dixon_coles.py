import numpy as np
import math

def calcular_poisson(lam, k):
    """Calcula la probabilidad de anotar 'k' goles con una expectativa 'lam'"""
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def dixon_coles_1x2(xg_local, xg_visita, rho=-0.15, max_goles=5):
    """
    Calcula las probabilidades de Local, Empate y Visita (1X2) 
    aplicando la corrección de Dixon-Coles para inflar los empates.
    """
    # 1. Crear matrices de probabilidad de Poisson pura
    prob_local = [calcular_poisson(xg_local, i) for i in range(max_goles + 1)]
    prob_visita = [calcular_poisson(xg_visita, j) for j in range(max_goles + 1)]
    
    matriz_prob = np.outer(prob_local, prob_visita)
    
    # 2. Aplicar el factor de corrección de Dixon-Coles (Tau)
    # Inflamos el 0-0 y el 1-1, y ajustamos el 1-0 y 0-1
    matriz_prob[0, 0] *= (1 - (xg_local * xg_visita * rho))
    matriz_prob[1, 0] *= (1 + (xg_local * rho))
    matriz_prob[0, 1] *= (1 + (xg_visita * rho))
    matriz_prob[1, 1] *= (1 - rho)
    
    # 3. Normalizar la matriz para que la suma total sea exactamente 1.0 (100%)
    matriz_prob = matriz_prob / np.sum(matriz_prob)
    
    # 4. Sumar las diagonales para obtener el 1X2 final
    prob_gana_local = np.tril(matriz_prob, -1).sum()
    prob_empate = np.trace(matriz_prob)
    prob_gana_visita = np.triu(matriz_prob, 1).sum()
    
    return prob_gana_local, prob_empate, prob_gana_visita

if __name__ == "__main__":
    # Prueba del algoritmo: Supongamos un partido muy parejo (ej. América vs Tigres)
    xg_home = 1.30
    xg_away = 1.25
    
    p_local, p_empate, p_visita = dixon_coles_1x2(xg_home, xg_away)
    
    print("\n--- PRUEBA DEL MOTOR DIXON-COLES ---")
    print(f"Expectativa de Goles (xG): Local {xg_home} | Visita {xg_away}")
    print(f"Probabilidad Local:  {p_local:.2%}")
    print(f"Probabilidad Empate: {p_empate:.2%} (Inflada por Dixon-Coles)")
    print(f"Probabilidad Visita: {p_visita:.2%}")
    print("------------------------------------\n")