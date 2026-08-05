import numpy as np
import math
from collections import defaultdict

class Agent:
    def __init__(self, player: str, env_cfg: dict):
        """
        Inicializa el agente.
        - player: 'player_0' o 'player_1'
        - env_cfg: configuración del entorno (mapa, parámetros, etc.)
        """
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"
        self.env_cfg = env_cfg
        
        # Parámetros del agente
        self.relic_node_positions = []
        self.relic_nodes_discovered = set()
        self.relic_tiles_scored = {}  # (x, y) -> score
        self.energy_node_positions = []
        self.map_explored = np.zeros((24, 24), dtype=bool)
        self.opponent_positions = []
        self.match_number = 0
        self.param_exploration = True  # Fase de exploración en partida 1
        
        # Historial de acciones
        self.action_history = []
        
        # Parámetros estimados (se descubren jugando)
        self.estimated_unit_move_cost = 3
        self.estimated_unit_sap_range = 5
        self.estimated_unit_sap_cost = 40
        self.estimated_sensor_range = 3
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Devuelve las acciones para cada unidad en el paso actual.
        """
        actions = {}
        
        team_obs = obs["team_obs"][self.player]
        units = team_obs["units"]
        unit_mask = team_obs["unit_mask"]
        sensor_mask = team_obs["sensor_mask"]
        relic_nodes = obs["relic_nodes"]
        relic_nodes_mask = obs["relic_nodes_mask"]
        energy_nodes = obs["energy_nodes"]
        energy_nodes_mask = obs["energy_nodes_mask"]
        map_features = obs["map_features"]
        
        # Actualizar conocimiento del mapa
        self._update_knowledge(obs)
        
        # Determinar si estamos en fase de exploración o explotación
        if self.match_number == 0 and step < 50:
            self.param_exploration = True
        else:
            self.param_exploration = False
            
        # Para cada unidad, decidir acción
        for unit_id in range(len(units)):
            if not unit_mask[unit_id]:
                continue
                
            unit = units[unit_id]
            pos = (unit["x"], unit["y"])
            energy = unit["energy"]
            
            if energy < 30:
                action = self._action_recharge(unit, obs)
            elif self.param_exploration:
                action = self._action_explore(unit, obs)
            else:
                action = self._action_exploit(unit, obs)
                
            actions[unit_id] = action
            
        self.action_history.append(actions)
        return actions
    
    def _update_knowledge(self, obs):
        """Actualiza el conocimiento del agente sobre el mapa y los nodos."""
        team_obs = obs["team_obs"][self.player]
        sensor_mask = team_obs["sensor_mask"]
        
        visible = np.where(sensor_mask > 0)
        for y, x in zip(visible[0], visible[1]):
            if 0 <= x < 24 and 0 <= y < 24:
                self.map_explored[y, x] = True
                
        relic_nodes = obs["relic_nodes"]
        relic_mask = obs["relic_nodes_mask"]
        for i in range(len(relic_nodes)):
            if relic_mask[i]:
                pos = (relic_nodes[i]["x"], relic_nodes[i]["y"])
                if pos not in self.relic_nodes_discovered:
                    self.relic_nodes_discovered.add(pos)
                    self.relic_node_positions.append(pos)
                    
        energy_nodes = obs["energy_nodes"]
        energy_mask = obs["energy_nodes_mask"]
        for i in range(len(energy_nodes)):
            if energy_mask[i]:
                pos = (energy_nodes[i]["x"], energy_nodes[i]["y"])
                if pos not in self.energy_node_positions:
                    self.energy_node_positions.append(pos)
                    
    def _action_recharge(self, unit, obs):
        """Acción para recargar energía."""
        pos = (unit["x"], unit["y"])
        energy = unit["energy"]
        
        if pos in self.energy_node_positions:
            return [0, 0, 0, 0]
            
        if self.energy_node_positions:
            closest = min(self.energy_node_positions, 
                         key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
            dx = np.clip(closest[0] - pos[0], -1, 1)
            dy = np.clip(closest[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
            
        return self._action_explore(unit, obs)
    
    def _action_explore(self, unit, obs):
        """Acción de exploración: moverse hacia áreas no exploradas."""
        pos = (unit["x"], unit["y"])
        
        unexplored = np.where(~self.map_explored)
        if len(unexplored[0]) > 0:
            min_dist = float('inf')
            target = None
            for y, x in zip(unexplored[0], unexplored[1]):
                dist = abs(x - pos[0]) + abs(y - pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target = (x, y)
                    
            if target:
                dx = np.clip(target[0] - pos[0], -1, 1)
                dy = np.clip(target[1] - pos[1], -1, 1)
                return [dx, dy, 0, 0]
                
        if self.relic_node_positions:
            target = min(self.relic_node_positions, 
                        key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
            dx = np.clip(target[0] - pos[0], -1, 1)
            dy = np.clip(target[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
            
        return [0, 0, 0, 0]
    
    def _action_exploit(self, unit, obs):
        """Acción de explotación: maximizar puntos y atacar al oponente."""
        pos = (unit["x"], unit["y"])
        energy = unit["energy"]
        
        opponent_units = obs["team_obs"][self.opponent]["units"]
        opponent_mask = obs["team_obs"][self.opponent]["unit_mask"]
        
        # 1. Ir a nodos de reliquia
        if self.relic_node_positions:
            sorted_relics = sorted(self.relic_node_positions, 
                                  key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
            
            for relic_pos in sorted_relics:
                enemy_near = False
                for i in range(len(opponent_units)):
                    if opponent_mask[i]:
                        opp_pos = (opponent_units[i]["x"], opponent_units[i]["y"])
                        if abs(opp_pos[0] - relic_pos[0]) + abs(opp_pos[1] - relic_pos[1]) <= 3:
                            enemy_near = True
                            break
                            
                if not enemy_near or energy > 80:
                    dx = np.clip(relic_pos[0] - pos[0], -1, 1)
                    dy = np.clip(relic_pos[1] - pos[1], -1, 1)
                    return [dx, dy, 0, 0]
                    
        # 2. Atacar enemigos cercanos
        for i in range(len(opponent_units)):
            if opponent_mask[i]:
                opp_pos = (opponent_units[i]["x"], opponent_units[i]["y"])
                dist = abs(opp_pos[0] - pos[0]) + abs(opp_pos[1] - pos[1])
                
                if dist <= self.estimated_unit_sap_range and energy >= self.estimated_unit_sap_cost:
                    dx_sap = opp_pos[0] - pos[0]
                    dy_sap = opp_pos[1] - pos[1]
                    return [0, 0, dx_sap, dy_sap]
                    
                elif dist <= self.estimated_unit_sap_range + 2 and energy >= 50:
                    dx = np.clip(opp_pos[0] - pos[0], -1, 1)
                    dy = np.clip(opp_pos[1] - pos[1], -1, 1)
                    return [dx, dy, 0, 0]
                    
        # 3. Moverse al centro
        center = (12, 12)
        dx = np.clip(center[0] - pos[0], -1, 1)
        dy = np.clip(center[1] - pos[1], -1, 1)
        return [dx, dy, 0, 0]
    
    def reset(self):
        """Reinicia el estado del agente entre partidas."""
        self.action_history = []
        self.match_number += 1
        
        if self.match_number >= 1:
            self.param_exploration = False