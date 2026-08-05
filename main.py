import json
import sys
from agent import Agent

# Instancia global del agente (se crea al cargar)
_agent = None

def agent(obs):
    global _agent
    if _agent is None:
        # Inferir jugador y configuración desde obs
        player = obs.get("player", 0)
        env_cfg = {}  # no tenemos acceso directo
        _agent = Agent(player, env_cfg)
    return _agent.act(obs)

# Si se ejecuta como script, no hacemos nada (Kaggle importa la función)
if __name__ == "__main__":
    pass
