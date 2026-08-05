import json
import sys
from agent import Agent

def main():
    data = sys.stdin.read()
    if not data:
        return
    
    config = json.loads(data)
    player = config.get("player", 0)
    env_cfg = config.get("env_cfg", {})
    
    agent = Agent(player, env_cfg)
    
    # El entorno llama a act() en cada paso
    # La función act recibe obs y devuelve acciones
    # Pero en Kaggriculture, la función se llama con obs
    # y se espera un dict con "farmer", "hands", "market"
    
    # Simplemente llamamos a agent.act(obs) en cada step
    # El bucle lo maneja el entorno
    # Esta función main es para compatibilidad con el formato de Kaggle
    # En realidad, el entorno espera una función que recibe obs y devuelve actions
    
    # Para este agente, definimos una función wrapper
    def agent_fn(obs):
        return agent.act(obs)
    
    # El entorno ejecutará agent_fn directamente
    # Así que retornamos la función
    return agent_fn

if __name__ == "__main__":
    # Si se ejecuta como script, se espera que la función principal retorne el agente
    agent_fn = main()
    # No hacemos nada más; el entorno llamará a agent_fn
    pass
