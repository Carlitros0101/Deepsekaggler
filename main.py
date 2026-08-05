import json
import sys
from agent import Agent

def main():
    data = sys.stdin.read()
    if not data:
        return
        
    config = json.loads(data)
    
    player = config.get("player", "player_0")
    env_cfg = config.get("env_cfg", {})
    steps = config.get("steps", [])
    
    agent = Agent(player, env_cfg)
    
    for step_data in steps:
        step = step_data.get("step", 0)
        obs = step_data.get("obs", {})
        remaining = step_data.get("remainingOverageTime", 60)
        
        if step == 0:
            agent.reset()
            
        actions = agent.act(step, obs, remaining)
        
        print(json.dumps(actions))
        sys.stdout.flush()

if __name__ == "__main__":
    main()