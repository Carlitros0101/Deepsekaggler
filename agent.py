import numpy as np
import math
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

class Agent:
    def __init__(self, player: str, env_cfg: dict):
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"
        self.env_cfg = env_cfg
        
        # Parámetros del juego (se estiman dinámicamente)
        self.params = {
            "unit_move_cost": 3.0,
            "unit_sap_range": 5,
            "unit_sap_cost": 40,
            "unit_sensor_range": 3,
            "relic_score_radius": 2,  # estimado
        }
        
        # Conocimiento del mapa
        self.map_size = 24
        self.map_explored = np.zeros((self.map_size, self.map_size), dtype=bool)
        self.map_visited = np.zeros((self.map_size, self.map_size), dtype=int)  # contador de visitas
        
        # Nodos conocidos
        self.relic_nodes = []          # lista de (x, y)
        self.relic_tiles = {}          # (x, y) -> score acumulado (para descubrir tiles que dan puntos)
        self.energy_nodes = []         # lista de (x, y)
        
        # Unidades enemigas observadas (histórico)
        self.enemy_positions_history = deque(maxlen=20)
        self.enemy_energy_history = deque(maxlen=20)
        
        # Estado interno
        self.match_number = 0
        self.step = 0
        self.unit_roles = {}  # unit_id -> "explorer", "attacker", "supporter"
        self.unit_targets = {}  # unit_id -> (x, y)
        
        # Exploración por frontera
        self.frontier_cells = []
        
        # Sistema de votación para objetivos
        self.relic_priority = {}  # (x, y) -> prioridad (mayor = mejor)
        
        # Últimas acciones
        self.last_actions = {}
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.step = step
        actions = {}
        
        # 1. Actualizar todo el conocimiento
        self._update_knowledge(obs)
        
        # 2. Estimar parámetros del juego
        self._estimate_params(obs)
        
        # 3. Decidir roles de las unidades (solo al inicio)
        team_obs = obs["team_obs"][self.player]
        units = team_obs["units"]
        unit_mask = team_obs["unit_mask"]
        num_units = sum(unit_mask)
        if step % 10 == 0 or not self.unit_roles:
            self._assign_roles(units, unit_mask, num_units)
        
        # 4. Calcular prioridades de los nodos de reliquia
        self._compute_relic_priorities(obs)
        
        # 5. Para cada unidad, decidir acción
        for unit_id in range(len(units)):
            if not unit_mask[unit_id]:
                continue
            unit = units[unit_id]
            pos = (unit["x"], unit["y"])
            energy = unit["energy"]
            
            # Obtener rol
            role = self.unit_roles.get(unit_id, "explorer")
            
            # Decidir acción según rol y situación
            if energy < 25:
                action = self._action_recharge(unit, obs)
            elif role == "explorer" and self.match_number == 0:
                action = self._action_explore_frontier(unit, obs)
            elif role == "attacker":
                action = self._action_attack(unit, obs)
            else:  # supporter o explotación
                action = self._action_harvest(unit, obs)
            
            # Asegurar que la acción sea válida (dentro de [-1,1] y sap dentro de [-24,24])
            action = self._sanitize_action(action)
            actions[unit_id] = action
            self.last_actions[unit_id] = action
        
        return actions
    
    def _update_knowledge(self, obs):
        """Actualiza mapa explorado, nodos y enemigos."""
        team_obs = obs["team_obs"][self.player]
        sensor_mask = team_obs["sensor_mask"]
        
        # Casillas visibles
        visible = np.where(sensor_mask > 0)
        for y, x in zip(visible[0], visible[1]):
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.map_explored[y, x] = True
                self.map_visited[y, x] += 1
        
        # Nodos de reliquia
        relic_nodes = obs["relic_nodes"]
        relic_mask = obs["relic_nodes_mask"]
        for i in range(len(relic_nodes)):
            if relic_mask[i]:
                pos = (relic_nodes[i]["x"], relic_nodes[i]["y"])
                if pos not in self.relic_nodes:
                    self.relic_nodes.append(pos)
        
        # Nodos de energía
        energy_nodes = obs["energy_nodes"]
        energy_mask = obs["energy_nodes_mask"]
        for i in range(len(energy_nodes)):
            if energy_mask[i]:
                pos = (energy_nodes[i]["x"], energy_nodes[i]["y"])
                if pos not in self.energy_nodes:
                    self.energy_nodes.append(pos)
        
        # Detectar tiles de reliquia que dan puntos (cuando la puntuación aumenta)
        # No tenemos acceso directo a la puntuación, pero podemos inferir si estamos en un tile que da puntos
        # al observar que el marcador de reliquia cambia. Como no se da explícitamente, usaremos heurística:
        # si estamos cerca de un nodo de reliquia y el equipo obtiene puntos, lo registramos.
        # (En la práctica, se puede leer el score de la observación, pero no está en el API público)
        # En su lugar, marcaremos todas las casillas alrededor de los nodos como potenciales.
        for rx, ry in self.relic_nodes:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    tx, ty = rx + dx, ry + dy
                    if 0 <= tx < self.map_size and 0 <= ty < self.map_size:
                        if (tx, ty) not in self.relic_tiles:
                            self.relic_tiles[(tx, ty)] = 0
        
        # Enemigos visibles
        opponent_obs = obs["team_obs"][self.opponent]
        opp_units = opponent_obs["units"]
        opp_mask = opponent_obs["unit_mask"]
        for i in range(len(opp_units)):
            if opp_mask[i]:
                pos = (opp_units[i]["x"], opp_units[i]["y"])
                self.enemy_positions_history.append(pos)
                self.enemy_energy_history.append(opp_units[i]["energy"])
        
        # Actualizar frontera (casillas exploradas con vecinos no explorados)
        self._update_frontier()
    
    def _update_frontier(self):
        """Encuentra las casillas de frontera para exploración."""
        frontier = []
        for y in range(self.map_size):
            for x in range(self.map_size):
                if self.map_explored[y, x]:
                    # Revisar vecinos
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                            if not self.map_explored[ny, nx]:
                                frontier.append((x, y))
                                break
        self.frontier_cells = frontier
    
    def _estimate_params(self, obs):
        """Estima parámetros del juego observando acciones y cambios de energía."""
        team_obs = obs["team_obs"][self.player]
        units = team_obs["units"]
        unit_mask = team_obs["unit_mask"]
        
        # Estimar coste de movimiento: comparar energía entre pasos
        # (esto requiere guardar energía previa, lo hacemos en un buffer)
        if not hasattr(self, 'prev_energy'):
            self.prev_energy = {}
        
        for i in range(len(units)):
            if unit_mask[i]:
                uid = i
                energy = units[i]["energy"]
                if uid in self.prev_energy:
                    delta = self.prev_energy[uid] - energy
                    # Si delta > 0 y la unidad se movió, es el coste de movimiento
                    # Pero puede incluir recarga o daño. Asumimos que si se movió, coste = delta
                    if uid in self.last_actions and self.last_actions[uid][0:2] != [0,0]:
                        if delta > 0:
                            self.params["unit_move_cost"] = 0.9 * self.params["unit_move_cost"] + 0.1 * delta
                self.prev_energy[uid] = energy
        
        # Estimar rango de sap viendo ataques enemigos (enemigos atacan a distancia)
        # No tenemos acceso a las acciones enemigas, pero podemos inferir rango por distancia a la que nos atacan
        # (esto es complejo, lo dejamos como está)
        # También podemos usar la configuración del entorno si está disponible
        if "unit_sap_range" in self.env_cfg:
            self.params["unit_sap_range"] = self.env_cfg["unit_sap_range"]
        if "unit_sap_cost" in self.env_cfg:
            self.params["unit_sap_cost"] = self.env_cfg["unit_sap_cost"]
        if "unit_move_cost" in self.env_cfg:
            self.params["unit_move_cost"] = self.env_cfg["unit_move_cost"]
    
    def _assign_roles(self, units, unit_mask, num_units):
        """Asigna roles a las unidades según el número y la fase del juego."""
        self.unit_roles = {}
        unit_ids = [i for i in range(len(units)) if unit_mask[i]]
        
        if num_units == 0:
            return
        
        # En partida 1: todos exploradores
        if self.match_number == 0:
            for uid in unit_ids:
                self.unit_roles[uid] = "explorer"
            return
        
        # En partidas posteriores: mezcla de atacantes y recolectores
        # Ordenar por energía (los que tienen más energía son atacantes)
        sorted_units = sorted(unit_ids, key=lambda uid: units[uid]["energy"], reverse=True)
        
        # Los 2 con más energía son atacantes, el resto son recolectores
        num_attackers = min(2, num_units)
        for i, uid in enumerate(sorted_units):
            if i < num_attackers:
                self.unit_roles[uid] = "attacker"
            else:
                self.unit_roles[uid] = "supporter"
    
    def _compute_relic_priorities(self, obs):
        """Calcula prioridad de cada nodo de reliquia (distancia, presencia enemiga)."""
        team_obs = obs["team_obs"][self.player]
        units = team_obs["units"]
        unit_mask = team_obs["unit_mask"]
        
        # Calcular centro de masa de nuestras unidades
        our_positions = []
        for i in range(len(units)):
            if unit_mask[i]:
                our_positions.append((units[i]["x"], units[i]["y"]))
        if not our_positions:
            return
        
        center_x = np.mean([p[0] for p in our_positions])
        center_y = np.mean([p[1] for p in our_positions])
        
        # Enemigos visibles
        opponent_obs = obs["team_obs"][self.opponent]
        opp_units = opponent_obs["units"]
        opp_mask = opponent_obs["unit_mask"]
        enemy_positions = []
        for i in range(len(opp_units)):
            if opp_mask[i]:
                enemy_positions.append((opp_units[i]["x"], opp_units[i]["y"]))
        
        # Calcular prioridad de cada nodo de reliquia
        for relic in self.relic_nodes:
            dist_to_us = abs(relic[0] - center_x) + abs(relic[1] - center_y)
            # Distancia al enemigo más cercano
            if enemy_positions:
                min_enemy_dist = min(abs(relic[0] - ex) + abs(relic[1] - ey) for ex, ey in enemy_positions)
            else:
                min_enemy_dist = 100
            
            # Prioridad: queremos nodos cercanos a nosotros y lejos del enemigo
            priority = -dist_to_us + 0.5 * min_enemy_dist
            self.relic_priority[relic] = priority
    
    def _action_recharge(self, unit, obs):
        """Recarga energía yendo al nodo de energía más cercano."""
        pos = (unit["x"], unit["y"])
        if pos in self.energy_nodes:
            return [0, 0, 0, 0]  # ya está cargando
        
        if self.energy_nodes:
            # Elegir el nodo de energía más cercano
            closest = min(self.energy_nodes, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))
            dx = np.clip(closest[0] - pos[0], -1, 1)
            dy = np.clip(closest[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
        else:
            # Si no hay nodos de energía conocidos, explorar
            return self._action_explore_frontier(unit, obs)
    
    def _action_explore_frontier(self, unit, obs):
        """Se mueve hacia la frontera más cercana."""
        pos = (unit["x"], unit["y"])
        if not self.frontier_cells:
            # Si no hay frontera, ir al centro
            center = (12, 12)
            dx = np.clip(center[0] - pos[0], -1, 1)
            dy = np.clip(center[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
        
        # Encontrar la frontera más cercana (con menos visitas)
        best = None
        best_score = float('inf')
        for fx, fy in self.frontier_cells:
            dist = abs(fx - pos[0]) + abs(fy - pos[1])
            # Penalizar casillas ya visitadas
            visits = self.map_visited[fy, fx]
            score = dist + 0.5 * visits
            if score < best_score:
                best_score = score
                best = (fx, fy)
        
        if best:
            dx = np.clip(best[0] - pos[0], -1, 1)
            dy = np.clip(best[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
        return [0, 0, 0, 0]
    
    def _action_attack(self, unit, obs):
        """Estrategia de ataque: persigue y ataca enemigos vulnerables."""
        pos = (unit["x"], unit["y"])
        energy = unit["energy"]
        
        opponent_obs = obs["team_obs"][self.opponent]
        opp_units = opponent_obs["units"]
        opp_mask = opponent_obs["unit_mask"]
        
        # Lista de enemigos visibles con su energía y distancia
        enemies = []
        for i in range(len(opp_units)):
            if opp_mask[i]:
                enemy_pos = (opp_units[i]["x"], opp_units[i]["y"])
                enemy_energy = opp_units[i]["energy"]
                dist = abs(enemy_pos[0] - pos[0]) + abs(enemy_pos[1] - pos[1])
                enemies.append((enemy_pos, enemy_energy, dist))
        
        # Ordenar enemigos por: bajo energy, cerca
        enemies.sort(key=lambda e: (e[1], e[2]))
        
        for enemy_pos, enemy_energy, dist in enemies:
            # Si el enemigo está en rango de sap y tenemos suficiente energía
            sap_cost = self.params["unit_sap_cost"]
            sap_range = self.params["unit_sap_range"]
            
            if dist <= sap_range and energy >= sap_cost:
                # Atacar
                dx_sap = enemy_pos[0] - pos[0]
                dy_sap = enemy_pos[1] - pos[1]
                return [0, 0, dx_sap, dy_sap]
            
            # Si el enemigo está a 1-2 pasos de distancia, acercarse para atacar
            elif dist <= sap_range + 2 and energy >= 50:
                dx = np.clip(enemy_pos[0] - pos[0], -1, 1)
                dy = np.clip(enemy_pos[1] - pos[1], -1, 1)
                return [dx, dy, 0, 0]
        
        # Si no hay enemigos cercanos, ir a un nodo de reliquia
        return self._action_harvest(unit, obs)
    
    def _action_harvest(self, unit, obs):
        """Recolecta puntos en nodos de reliquia con mayor prioridad."""
        pos = (unit["x"], unit["y"])
        
        if not self.relic_nodes:
            # Si no hay nodos, explorar
            return self._action_explore_frontier(unit, obs)
        
        # Elegir el nodo de reliquia con mayor prioridad (más cercano y menos disputado)
        best_relic = None
        best_score = -float('inf')
        for relic in self.relic_nodes:
            priority = self.relic_priority.get(relic, 0)
            dist = abs(relic[0] - pos[0]) + abs(relic[1] - pos[1])
            # Preferir nodos cercanos y con alta prioridad
            score = priority - 0.5 * dist
            if score > best_score:
                best_score = score
                best_relic = relic
        
        if best_relic:
            dx = np.clip(best_relic[0] - pos[0], -1, 1)
            dy = np.clip(best_relic[1] - pos[1], -1, 1)
            return [dx, dy, 0, 0]
        
        return [0, 0, 0, 0]
    
    def _sanitize_action(self, action):
        """Asegura que los valores estén en el rango permitido."""
        # Movimiento: -1, 0, 1
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], -1, 1)
        # Sap: puede ser cualquier delta, pero limitamos a [-24,24] para evitar errores
        if len(action) >= 4:
            action[2] = np.clip(action[2], -24, 24)
            action[3] = np.clip(action[3], -24, 24)
        return action
    
    def reset(self):
        """Reinicia el estado entre partidas (conserva conocimiento)."""
        self.match_number += 1
        self.step = 0
        self.unit_roles = {}
        self.unit_targets = {}
        self.last_actions = {}
        self.prev_energy = {}
        # No reiniciamos el mapa ni los nodos (se mantienen)
        # La exploración se reinicia en partida 1, pero en partidas 2-5 ya tenemos el mapa