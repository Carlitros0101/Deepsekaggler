import json
import sys
from agent import Agent

# Instancia global (se crea al primer llamado)
_agent = None

def agent(obs):
    global _agent
    if _agent is None:
        player = obs.get("player", 0)
        env_cfg = {}  # no tenemos acceso directo
        _agent = Agent(player, env_cfg)
    return _agent.act(obs)

if __name__ == "__main__":
    pass
