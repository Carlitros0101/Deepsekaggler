#!/usr/bin/env python3
"""
Script para probar el agente localmente usando kaggle_environments.
Ejecuta una partida y muestra el resultado.
"""

import json
import sys
import os
from kaggle_environments import make
from agent import Agent

def run_game(agent_func, opponent="random", steps=720, verbose=True):
    """Ejecuta una partida y devuelve el resultado."""
    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)
    
    # Si el oponente es "self", jugamos contra una copia de nosotros mismos
    if opponent == "self":
        opponent = agent_func
    
    env.run([agent_func, opponent])
    
    # Obtener resultados
    final = env.steps[-1]
    money = [s.reward for s in final]
    status = [s.status for s in final]
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"RESULTADO DE LA PARTIDA")
        print(f"{'='*50}")
        print(f"Jugador 0 (nosotros): ${money[0]:.2f}  -  {status[0]}")
        print(f"Jugador 1 (oponente): ${money[1]:.2f}  -  {status[1]}")
        print(f"Ganador: {'Nosotros' if money[0] > money[1] else 'Oponente' if money[1] > money[0] else 'Empate'}")
        print(f"{'='*50}\n")
    
    return env, money

def test_agent():
    """Prueba el agente contra varios oponentes."""
    # Crear una instancia del agente
    agent_instance = Agent(player=0)
    
    # Función wrapper para el agente
    def agent_wrapper(obs):
        return agent_instance.act(obs)
    
    # Probar contra oponentes
    opponents = ["random", "pass", "self"]
    results = []
    
    for opp in opponents:
        print(f"\n▶ Probando contra '{opp}'...")
        env, money = run_game(agent_wrapper, opponent=opp, steps=200, verbose=True)
        results.append({
            "opponent": opp,
            "our_money": money[0],
            "their_money": money[1],
            "win": money[0] > money[1]
        })
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    for r in results:
        print(f"Contra {r['opponent']:>6}  →  Nosotros: ${r['our_money']:.2f}  Oponente: ${r['their_money']:.2f}  {'✅ GANAMOS' if r['win'] else '❌ PERDEMOS'}")
    print("="*60)
    
    # Guardar estadísticas
    agent_instance._save_stats()
    print("\n📊 Estadísticas guardadas en stats.json")

def visualize_game():
    """Ejecuta una partida y la muestra visualmente en el navegador."""
    env = make("kaggriculture", configuration={"episodeSteps": 200}, debug=True)
    agent_instance = Agent(player=0)
    def agent_wrapper(obs):
        return agent_instance.act(obs)
    env.run([agent_wrapper, "random"])
    # Renderizar en el navegador
    env.render(mode="ipython", width=1000, height=700)
    print("✅ Visualización abierta. Si estás en un notebook, verás el render.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prueba local del agente Kaggriculture")
    parser.add_argument("--steps", type=int, default=200, help="Número de pasos (días*24)")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "pass", "self"], help="Oponente")
    parser.add_argument("--visualize", action="store_true", help="Abrir visualización (solo en notebook)")
    parser.add_argument("--save", action="store_true", help="Guardar replay en replay.json")
    args = parser.parse_args()
    
    # Crear agente
    agent_instance = Agent(player=0)
    def agent_wrapper(obs):
        return agent_instance.act(obs)
    
    if args.visualize:
        visualize_game()
    else:
        env, money = run_game(agent_wrapper, opponent=args.opponent, steps=args.steps)
        if args.save:
            with open("replay.json", "w") as f:
                json.dump(env.toJSON(), f)
            print("📁 Replay guardado en replay.json")
