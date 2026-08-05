import numpy as np
import json
from collections import defaultdict

class Agent:
    def __init__(self, player, env_cfg):
        self.player = player
        self.env_cfg = env_cfg
        self.step = 0
        self.day = 0
        self.turn = 0
        
        # Estado interno
        self.farmer_pos = None
        self.hands_pos = []
        self.tiles = None
        self.money = 0
        self.shed = {}
        self.seeds = {}
        self.inventories = []
        
        # Estrategia
        self.phase = "early"  # early, mid, late
        self.crops_planted = 0
        self.animals_placed = 0
        self.quadrants_bought = 0
        
        # Planificación
        self.plants = {}  # (x,y) -> info
        self.animals = {}  # (x,y) -> info
        
        # Umbrales
        self.WHEAT_PRICE = 25
        self.CARROT_PRICE = 35
        self.TOMATO_PRICE = 60
        self.STRAWBERRY_PRICE = 120
        self.MELON_PRICE = 250
        
    def act(self, obs):
        """Acción principal del agente."""
        self.step = obs.get("step", 0)
        self.day = obs.get("day", 0)
        self.turn = obs.get("hour", 0)
        
        # Actualizar estado
        self._update_state(obs)
        
        # Determinar fase
        self._determine_phase()
        
        # Generar acciones
        farmer_actions = []
        market_actions = []
        
        # 1. Acciones de mercado (siempre primero)
        market_actions.extend(self._market_actions(obs))
        
        # 2. Acciones del agricultor principal
        farmer_actions = self._farmer_actions(obs)
        
        # 3. Acciones de los trabajadores (si hay)
        hand_actions = self._hand_actions(obs)
        
        return {
            "farmer": farmer_actions,
            "hands": hand_actions,
            "market": market_actions
        }
    
    def _update_state(self, obs):
        """Actualiza el conocimiento interno."""
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        
        self.farmer_pos = tuple(farm["farmer"])
        self.hands_pos = [tuple(h) for h in farm["hands"]]
        self.tiles = farm["tiles"]
        self.money = farm["money"]
        self.shed = private.get("shed", {})
        self.seeds = private.get("seeds", {})
        self.inventories = private.get("inventories", [])
    
    def _determine_phase(self):
        """Determina la fase del juego según el día."""
        if self.day <= 5:
            self.phase = "early"
        elif self.day <= 15:
            self.phase = "mid"
        else:
            self.phase = "late"
    
    def _market_actions(self, obs):
        """Acciones de mercado."""
        actions = []
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        market = obs["market"]
        
        # Estrategia de compra/venta según fase
        
        # 1. Comprar semillas de trigo al inicio
        if self.phase == "early":
            # Comprar semillas de trigo si no tenemos y hay dinero
            if private["seeds"].get("WHEAT", 0) < 5 and farm["money"] >= 50:
                actions.append(["BUY_SEED", "WHEAT", 5])
            # Comprar fertilizante si tenemos dinero extra
            if private["shed"].get("FERTILIZER", 0) < 3 and farm["money"] >= 300:
                actions.append(["BUY_PRODUCT", "FERTILIZER", 3])
        
        # 2. Fase media: diversificar
        elif self.phase == "mid":
            # Comprar semillas de zanahoria y tomate
            if private["seeds"].get("CARROT", 0) < 3 and farm["money"] >= 60:
                actions.append(["BUY_SEED", "CARROT", 3])
            if private["seeds"].get("TOMATO", 0) < 2 and farm["money"] >= 100:
                actions.append(["BUY_SEED", "TOMATO", 2])
            # Comprar animales (gansos) para huevos
            if private["shed"].get("GOOSE", 0) < 2 and farm["money"] >= 600:
                actions.append(["BUY_ANIMAL", "GOOSE", 2])
        
        # 3. Fase tardía: comprar tierras y más animales
        elif self.phase == "late":
            # Comprar tierras si hay dinero y no todas compradas
            if len(farm["unlocked_quadrants"]) < 4 and farm["money"] >= 1000:
                actions.append(["BUY_LAND"])
            # Comprar más animales si hay espacio
            if private["shed"].get("GOOSE", 0) < 4 and farm["money"] >= 1200:
                actions.append(["BUY_ANIMAL", "GOOSE", 2])
            if private["shed"].get("COW", 0) < 2 and farm["money"] >= 800:
                actions.append(["BUY_ANIMAL", "COW", 1])
        
        # 4. Vender productos estratégicamente
        # Vender trigo cuando el precio es bueno (>25)
        wheat_in_shed = private["shed"].get("WHEAT", 0)
        if wheat_in_shed > 0:
            price = market["prices"].get("WHEAT", 25)
            if price >= 30 or (self.phase == "late" and wheat_in_shed > 10):
                actions.append(["SELL", "WHEAT", wheat_in_shed])
        
        # Vender zanahorias
        carrot_in_shed = private["shed"].get("CARROT", 0)
        if carrot_in_shed > 0 and self.phase != "early":
            price = market["prices"].get("CARROT", 35)
            if price >= 40 or (self.phase == "late" and carrot_in_shed > 5):
                actions.append(["SELL", "CARROT", carrot_in_shed])
        
        # Vender tomates
        tomato_in_shed = private["shed"].get("TOMATO", 0)
        if tomato_in_shed > 0 and self.phase != "early":
            price = market["prices"].get("TOMATO", 60)
            if price >= 70 or (self.phase == "late" and tomato_in_shed > 3):
                actions.append(["SELL", "TOMATO", tomato_in_shed])
        
        # Vender huevos (si los hay)
        eggs_in_shed = private["shed"].get("EGG", 0)
        if eggs_in_shed > 0 and self.phase != "early":
            price = market["prices"].get("EGG", 50)
            if price >= 55 or self.phase == "late":
                actions.append(["SELL", "EGG", eggs_in_shed])
        
        # Vender fertilizante si tenemos demasiado
        fert_in_shed = private["shed"].get("FERTILIZER", 0)
        if fert_in_shed > 5 and self.phase == "late":
            actions.append(["SELL", "FERTILIZER", fert_in_shed - 5])
        
        # Contratar trabajadores (máximo 3-4 al día)
        if farm["hires_today"] < 3 and self.phase != "early":
            actions.append(["HIRE"])
        
        return actions
    
    def _farmer_actions(self, obs):
        """Acciones del agricultor principal."""
        actions = []
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        
        x, y = farm["farmer"]
        tile = farm["tiles"][y][x]
        
        # 1. Si estamos en un tile con planta, gestionarla
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            planted_day = tile.get("planted_day", self.day)
            age = self.day - planted_day
            
            # Fertilizar si es el momento adecuado
            if tile.get("fertilized_until_day", -1) < 0 and private["shed"].get("FERTILIZER", 0) > 0:
                if crop in ["WHEAT", "CARROT"] and age <= 2:
                    actions.append(["FERTILIZE"])
                    return {"farmer": actions, "hands": [], "market": []}
            
            # Regar si no se ha regado hoy
            if not tile.get("watered_today", False):
                actions.append(["WATER"])
                return {"farmer": actions, "hands": [], "market": []}
            
            # Cosechar si está listo
            yield_units = tile.get("yield_units", 0)
            if yield_units > 0:
                # Si es trigo o zanahoria y tienen suficiente rendimiento
                if crop in ["WHEAT", "CARROT"]:
                    if yield_units >= 3:  # mínimo aceptable
                        actions.append(["HARVEST"])
                        return {"farmer": actions, "hands": [], "market": []}
                elif crop in ["TOMATO", "STRAWBERRY"]:
                    if yield_units >= 2:
                        actions.append(["HARVEST"])
                        return {"farmer": actions, "hands": [], "market": []}
                elif crop == "MELON":
                    if yield_units >= 4:
                        actions.append(["HARVEST"])
                        return {"farmer": actions, "hands": [], "market": []}
        
        # 2. Si estamos en un tile con animal
        elif isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            animal = tile.get("animal")
            if animal:
                # Alimentar si no se ha alimentado hoy
                if not tile.get("fed_today", False):
                    # Necesitamos trigo para alimentar
                    if private["shed"].get("WHEAT", 0) > 0:
                        # Para alimentar, el agricultor debe tener trigo en inventario
                        # Como no podemos mover items del shed al inventario directamente,
                        # usamos PICKUP si estamos cerca del shed o simplemente PASS
                        # y esperamos que el trabajador lo haga.
                        # Por simplicidad, asumimos que los trabajadores alimentan.
                        pass
                # Cuidar si no se ha cuidado hoy
                if not tile.get("cared_today", False):
                    actions.append(["CARE"])
                    return {"farmer": actions, "hands": [], "market": []}
                # Recolectar productos
                if tile.get("yield_units", 0) > 0:
                    actions.append(["HARVEST"])
                    return {"farmer": actions, "hands": [], "market": []}
                # Recolectar fertilizante
                if tile.get("fertilizer_available", False):
                    actions.append(["COLLECT_FERTILIZER"])
                    return {"farmer": actions, "hands": [], "market": []}
        
        # 3. Si estamos en un tile vacío y tenemos semillas, plantar
        if tile is None:
            # Prioridad de cultivos según fase
            if self.phase == "early":
                if private["seeds"].get("WHEAT", 0) > 0:
                    actions.append(["PLANT", "WHEAT"])
                    return {"farmer": actions, "hands": [], "market": []}
            elif self.phase == "mid":
                # Mezcla de cultivos
                if private["seeds"].get("CARROT", 0) > 0:
                    actions.append(["PLANT", "CARROT"])
                    return {"farmer": actions, "hands": [], "market": []}
                elif private["seeds"].get("TOMATO", 0) > 0:
                    actions.append(["PLANT", "TOMATO"])
                    return {"farmer": actions, "hands": [], "market": []}
                elif private["seeds"].get("WHEAT", 0) > 0:
                    actions.append(["PLANT", "WHEAT"])
                    return {"farmer": actions, "hands": [], "market": []}
            else:  # late
                # Cultivos más rentables
                if private["seeds"].get("MELON", 0) > 0:
                    actions.append(["PLANT", "MELON"])
                    return {"farmer": actions, "hands": [], "market": []}
                elif private["seeds"].get("STRAWBERRY", 0) > 0:
                    actions.append(["PLANT", "STRAWBERRY"])
                    return {"farmer": actions, "hands": [], "market": []}
                elif private["seeds"].get("CARROT", 0) > 0:
                    actions.append(["PLANT", "CARROT"])
                    return {"farmer": actions, "hands": [], "market": []}
                elif private["seeds"].get("WHEAT", 0) > 0:
                    actions.append(["PLANT", "WHEAT"])
                    return {"farmer": actions, "hands": [], "market": []}
        
        # 4. Si tenemos animales en el shed, colocarlos en estructuras
        if private["shed"].get("GOOSE", 0) > 0:
            # Buscar un coop vacío cerca
            coop_pos = self._find_empty_structure("COOP")
            if coop_pos and self._is_adjacent_to_shed(x, y, coop_pos[0], coop_pos[1]):
                actions.append(["PLACE", "GOOSE", 1])
                return {"farmer": actions, "hands": [], "market": []}
        
        if private["shed"].get("COW", 0) > 0:
            pasture_pos = self._find_empty_structure("PASTURE")
            if pasture_pos and self._is_adjacent_to_shed(x, y, pasture_pos[0], pasture_pos[1]):
                actions.append(["PLACE", "COW", 1])
                return {"farmer": actions, "hands": [], "market": []}
        
        # 5. Si no hay nada que hacer, moverse a un tile útil
        # Buscar tile vacío más cercano
        target = self._find_nearest_empty_tile()
        if target:
            dx = target[0] - x
            dy = target[1] - y
            if dx < 0:
                actions.append(["MOVE", "WEST"])
            elif dx > 0:
                actions.append(["MOVE", "EAST"])
            elif dy < 0:
                actions.append(["MOVE", "NORTH"])
            elif dy > 0:
                actions.append(["MOVE", "SOUTH"])
            return {"farmer": actions, "hands": [], "market": []}
        
        # Si todo falla, PASS
        actions.append(["PASS"])
        return {"farmer": actions, "hands": [], "market": []}
    
    def _hand_actions(self, obs):
        """Acciones de los trabajadores."""
        hands_actions = []
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        
        for i, (hx, hy) in enumerate(farm["hands"]):
            tile = farm["tiles"][hy][hx]
            actions = []
            
            # Trabajadores se encargan de tareas repetitivas: regar, alimentar, cosechar
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                # Regar si no se ha regado hoy
                if not tile.get("watered_today", False):
                    actions.append(["WATER"])
                # Cosechar si tiene rendimiento
                elif tile.get("yield_units", 0) > 0:
                    actions.append(["HARVEST"])
                else:
                    actions.append(["PASS"])
            
            elif isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                animal = tile.get("animal")
                if animal:
                    # Recolectar productos
                    if tile.get("yield_units", 0) > 0:
                        actions.append(["HARVEST"])
                    # Recolectar fertilizante
                    elif tile.get("fertilizer_available", False):
                        actions.append(["COLLECT_FERTILIZER"])
                    # Cuidar si no se ha cuidado
                    elif not tile.get("cared_today", False):
                        actions.append(["CARE"])
                    else:
                        actions.append(["PASS"])
                else:
                    # Construir estructuras para animales
                    if private["shed"].get("GOOSE", 0) > 0:
                        actions.append(["BUILD_COOP"])
                    elif private["shed"].get("COW", 0) > 0:
                        actions.append(["BUILD_PASTURE"])
                    else:
                        actions.append(["PASS"])
            
            # Si el tile está vacío y tenemos semillas, plantar
            elif tile is None:
                if private["seeds"].get("WHEAT", 0) > 0:
                    actions.append(["PLANT", "WHEAT"])
                elif private["seeds"].get("CARROT", 0) > 0:
                    actions.append(["PLANT", "CARROT"])
                elif private["seeds"].get("TOMATO", 0) > 0:
                    actions.append(["PLANT", "TOMATO"])
                else:
                    actions.append(["PASS"])
            
            # Si hay una maleza, cavar
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                actions.append(["DIG"])
            
            else:
                actions.append(["PASS"])
            
            hands_actions.append(actions)
        
        return hands_actions
    
    def _find_empty_tile(self, farm_tiles):
        """Busca un tile vacío en la granja."""
        for y in range(len(farm_tiles)):
            for x in range(len(farm_tiles[y])):
                if farm_tiles[y][x] is None:
                    return (x, y)
        return None
    
    def _find_nearest_empty_tile(self):
        """Encuentra el tile vacío más cercano al agricultor."""
        x, y = self.farmer_pos
        best = None
        best_dist = float('inf')
        for yy in range(len(self.tiles)):
            for xx in range(len(self.tiles[yy])):
                if self.tiles[yy][xx] is None:
                    dist = abs(xx - x) + abs(yy - y)
                    if dist < best_dist:
                        best_dist = dist
                        best = (xx, yy)
        return best
    
    def _find_empty_structure(self, kind):
        """Encuentra una estructura vacía del tipo especificado."""
        for y in range(len(self.tiles)):
            for x in range(len(self.tiles[y])):
                tile = self.tiles[y][x]
                if isinstance(tile, dict) and tile.get("kind") == kind:
                    if tile.get("animal") is None:
                        return (x, y)
        return None
    
    def _is_adjacent_to_shed(self, x1, y1, x2, y2):
        """Verifica si dos posiciones son adyacentes."""
        return abs(x1 - x2) + abs(y1 - y2) == 1
