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
    steps = config.get("steps", [])
    
    agent = Agent(player, env_cfg)
    
    for step_data in steps:
        obs = step_data.get("obs", {})
        
        # El agente devuelve las acciones
        actions = agent.act(obs)
        
        # Enviar acciones como JSON
        print(json.dumps(actions))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
