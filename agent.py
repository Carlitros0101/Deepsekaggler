import numpy as np
import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any

class Agent:
    def __init__(self, player: int, env_cfg: dict = None):
        self.player = player
        self.env_cfg = env_cfg or {}
        
        # Parámetros fijos del juego
        self.CROPS = {
            "WHEAT":   {"seed_cost": 10,  "base_price": 25, "first_yield": 2, "max_yield": 6, "ongoing": False, "fertilize_boost": 2},
            "CARROT":  {"seed_cost": 20,  "base_price": 35, "first_yield": 2, "max_yield": 4, "ongoing": False, "fertilize_boost": 2},
            "TOMATO":  {"seed_cost": 50,  "base_price": 60, "first_yield": 8, "max_yield": 4, "ongoing": True,  "fertilize_boost": 1},
            "STRAWBERRY": {"seed_cost": 100, "base_price": 120, "first_yield": 10, "max_yield": 4, "ongoing": True, "fertilize_boost": 1},
            "MELON":   {"seed_cost": 80,  "base_price": 250, "first_yield": 10, "max_yield": 6, "ongoing": False, "fertilize_boost": 2},
        }
        self.ANIMALS = {
            "GOOSE": {"cost": 300, "base_price": 50, "yield_interval": 1, "max_held": 4, "care_effect": 1},
            "COW":   {"cost": 400, "base_price": 160, "yield_interval": 2, "max_held": 6, "care_effect": 1},
            "SHEEP": {"cost": 500, "base_price": 200, "yield_interval": 3, "max_held": 6, "care_effect": 1},
        }
        self.BUILDINGS = {"COOP": 0, "PASTURE": 0}
        
        # Historial de precios (para predicción)
        self.price_history = defaultdict(lambda: deque(maxlen=10))
        self.inventory_history = defaultdict(lambda: deque(maxlen=10))
        
        # Estado interno
        self.day = 0
        self.step = 0
        self.money = 0
        self.tiles = None
        self.shed = {}
        self.seeds = {}
        self.inventories = []
        self.farmer_pos = (0, 0)
        self.hands_pos = []
        self.unlocked_quadrants = set()
        self.hires_today = 0
        self.last_action = None
        
        # Planificación
        self.crop_plan = []  # lista de (crop, cantidad) para plantar
        self.animal_plan = [] # lista de (animal, cantidad) para comprar
        self.land_target = 0  # cuántos cuadrantes queremos comprar
        self.target_workers = 1  # número deseado de trabajadores
        
        # Ventas fraccionadas
        self.sell_buffer = {}  # producto -> cantidad a vender gradualmente
        self.sell_rate = {}    # producto -> unidades por turno
        
        # Estrategia
        self.phase = "early"  # early, mid, late
        self.crop_priority = []
        
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        """
        self._parse_obs(obs)
        self.day = obs["day"]
        self.step = obs.get("step", 0)
        
        # Actualizar historiales
        self._update_price_history(obs["market"])
        
        # Decidir estrategia según el día
        self._update_phase()
        
        # Acciones del agricultor (farmer)
        farmer_actions = self._farmer_actions()
        
        # Acciones de trabajadores (hands)
        hand_actions = []
        for i in range(len(self.hands_pos)):
            hand_actions.append(self._hand_action(i))
        
        # Acciones de mercado (market orders)
        market_actions = self._market_actions()
        
        return {
            "farmer": farmer_actions,
            "hands": hand_actions,
            "market": market_actions
        }
    
    # ======================== PARSING ========================
    def _parse_obs(self, obs):
        me = obs["farms"][self.player]
        self.money = me["money"]
        self.tiles = me["tiles"]
        self.farmer_pos = tuple(me["farmer"])
        self.hands_pos = [tuple(p) for p in me["hands"]]
        self.unlocked_quadrants = set(me["unlocked_quadrants"])
        self.hires_today = me.get("hires_today", 0)
        
        private = obs["private"]
        self.shed = private["shed"]
        self.seeds = private["seeds"]
        self.inventories = private.get("inventories", [])
        
    def _update_price_history(self, market):
        for product, price in market["prices"].items():
            self.price_history[product].append(price)
        for product, inv in market["inventory"].items():
            self.inventory_history[product].append(inv)
    
    def _update_phase(self):
        if self.day < 10:
            self.phase = "early"
        elif self.day < 20:
            self.phase = "mid"
        else:
            self.phase = "late"
    
    # ======================== PREDICCIÓN DE PRECIOS ========================
    def _predict_price(self, product: str, horizon: int = 1) -> float:
        """Predice precio futuro usando media móvil y tendencia."""
        hist = list(self.price_history[product])
        if len(hist) < 2:
            return self.CROPS.get(product, {}).get("base_price", 50)
        
        # Media móvil simple (últimos N)
        n = min(5, len(hist))
        avg = np.mean(list(hist)[-n:])
        # Tendencia
        if len(hist) >= 3:
            trend = (hist[-1] - hist[-3]) / 2
        else:
            trend = hist[-1] - hist[-2] if len(hist) >= 2 else 0
        
        pred = avg + trend * horizon
        return max(1, pred)  # precio mínimo $1
    
    def _is_price_good_to_sell(self, product: str) -> bool:
        """Decide si es buen momento para vender."""
        if product not in self.price_history or len(self.price_history[product]) < 3:
            return False
        
        prices = list(self.price_history[product])
        # Si el precio actual es mayor que el promedio de los últimos 5, vender
        avg = np.mean(prices[-5:])
        current = prices[-1]
        # Si está por encima de la media + 10% y no ha bajado en los últimos 2 pasos
        if current > avg * 1.1:
            # Verificar que no esté cayendo
            if len(prices) >= 2 and prices[-1] >= prices[-2]:
                return True
        return False
    
    def _is_price_good_to_buy_seed(self, crop: str) -> bool:
        """Decide si es buen momento para comprar semillas (precio bajo)."""
        # Las semillas tienen precio fijo, no dependen del mercado
        return True  # siempre comprar si necesitamos
    
    # ======================== OPTIMIZACIÓN DE CARTERA ========================
    def _compute_crop_profitability(self, crop: str) -> float:
        """Calcula rentabilidad esperada de un cultivo (ingreso - coste) / días."""
        info = self.CROPS[crop]
        seed_cost = info["seed_cost"]
        base_price = self._predict_price(crop) or info["base_price"]
        max_yield = info["max_yield"]
        time_to_max = info["first_yield"] + 2  # aprox. tiempo hasta cosecha
        if info["ongoing"]:
            # Cultivos continuos: estimamos 4 cosechas en el tiempo restante
            remaining_days = max(0, 30 - self.day - time_to_max)
            harvests = min(info["max_yield"], remaining_days // 2 + 1)
            total_yield = harvests * 2  # 2 unidades por cosecha aprox.
            time = time_to_max + harvests * 2
        else:
            total_yield = max_yield
            time = time_to_max
        
        # Ajustar por fertilizante (opcional)
        expected_yield = total_yield * 1.2  # factor de mejora por riego
        
        revenue = expected_yield * base_price
        cost = seed_cost
        profit = revenue - cost
        if time <= 0:
            return 0
        return profit / time
    
    def _compute_animal_profitability(self, animal: str) -> float:
        """Calcula rentabilidad esperada de un animal (ingreso - coste) / día."""
        info = self.ANIMALS[animal]
        cost = info["cost"]
        base_price = self._predict_price(animal.upper()) or info["base_price"]
        interval = info["yield_interval"]
        max_held = info["max_held"]
        
        # Animal produce cada N días, con cuidado extra mejora
        remaining_days = max(0, 30 - self.day - 2)  # 2 días para construir
        harvests = remaining_days // interval
        total_yield = min(harvests * 2, max_held * 2)  # estimación conservadora
        revenue = total_yield * base_price
        # Coste de alimentación: 1 trigo por día (precio trigo)
        wheat_price = self._predict_price("WHEAT")
        feed_cost = remaining_days * wheat_price
        profit = revenue - cost - feed_cost
        return profit / max(1, remaining_days)
    
    def _select_best_crops(self, num_slots: int) -> List[str]:
        """Selecciona los cultivos más rentables para plantar."""
        crops = list(self.CROPS.keys())
        profits = [(c, self._compute_crop_profitability(c)) for c in crops]
        profits.sort(key=lambda x: x[1], reverse=True)
        # Siempre incluir trigo (para alimentar animales)
        if "WHEAT" not in [p[0] for p in profits[:2]]:
            # Asegurar al menos trigo
            return ["WHEAT"] + [p[0] for p in profits if p[0] != "WHEAT"][:num_slots-1]
        return [p[0] for p in profits[:num_slots]]
    
    def _select_best_animals(self) -> List[str]:
        """Selecciona los animales más rentables."""
        animals = list(self.ANIMALS.keys())
        profits = [(a, self._compute_animal_profitability(a)) for a in animals]
        profits.sort(key=lambda x: x[1], reverse=True)
        # Solo los mejores 1-2
        return [p[0] for p in profits[:2] if p[1] > 0]
    
    # ======================== ACCIONES DEL AGRICULTOR ========================
    def _farmer_actions(self) -> List[str]:
        """Decide la acción del agricultor principal."""
        x, y = self.farmer_pos
        tile = self.tiles[y][x] if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]) else None
        
        # Si hay algo para cosechar, priorizar
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
        
        # Si hay un animal con producto, cosechar
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            # Si tiene fertilizante disponible, recogerlo
            if tile.get("fertilizer_available", False):
                return ["COLLECT_FERTILIZER"]
        
        # Si estamos en una parcela vacía y tenemos semillas, plantar
        if tile is None and self.seeds:
            # Elegir la mejor semilla disponible
            for crop in self._select_best_crops(3):
                if self.seeds.get(crop, 0) > 0:
                    return ["PLANT", crop]
        
        # Si hay una planta sin regar hoy, regar
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not tile.get("watered_today", True):
                return ["WATER"]
            # Si está fertilizado, no regar (ya regado)
        
        # Si hay un animal sin alimentar hoy, alimentar
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            animal = tile.get("animal")
            if animal and not tile.get("fed_today", True):
                if self.shed.get("WHEAT", 0) > 0:
                    return ["FEED"]
            # Si podemos cuidar al animal (care)
            if animal and not tile.get("cared_today", True):
                return ["CARE"]
        
        # Si tenemos fertilizante y una planta, fertilizar
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if self.shed.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < self.day:
                return ["FERTILIZE"]
        
        # Si hay una estructura vacía y tenemos animal, colocar
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            animal_type = None
            if tile["kind"] == "COOP" and self.shed.get("GOOSE", 0) > 0:
                animal_type = "GOOSE"
            elif tile["kind"] == "PASTURE":
                if self.shed.get("COW", 0) > 0:
                    animal_type = "COW"
                elif self.shed.get("SHEEP", 0) > 0:
                    animal_type = "SHEEP"
            if animal_type:
                return ["PLACE", animal_type, 1]
        
        # Si no hay nada que hacer, moverse a un lugar útil
        return self._move_to_useful_tile()
    
    def _move_to_useful_tile(self) -> List[str]:
        """Se mueve hacia una parcela vacía o una estructura con trabajo."""
        # Buscar tile vacío más cercano o con planta sin regar o animal sin alimentar
        target = None
        best_score = -1
        
        for y in range(len(self.tiles)):
            for x in range(len(self.tiles[y])):
                tile = self.tiles[y][x]
                if tile is None:
                    dist = abs(x - self.farmer_pos[0]) + abs(y - self.farmer_pos[1])
                    score = -dist
                    if score > best_score:
                        best_score = score
                        target = (x, y)
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    if not tile.get("watered_today", True) or tile.get("yield_units", 0) > 0:
                        dist = abs(x - self.farmer_pos[0]) + abs(y - self.farmer_pos[1])
                        score = -dist + 10  # priorizar plantas
                        if score > best_score:
                            best_score = score
                            target = (x, y)
                elif isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                    if tile.get("animal") and not tile.get("fed_today", True):
                        dist = abs(x - self.farmer_pos[0]) + abs(y - self.farmer_pos[1])
                        score = -dist + 5
                        if score > best_score:
                            best_score = score
                            target = (x, y)
        
        if target:
            dx = np.clip(target[0] - self.farmer_pos[0], -1, 1)
            dy = np.clip(target[1] - self.farmer_pos[1], -1, 1)
            if dx == 0 and dy == 0:
                return ["PASS"]
            if dx == 1: return ["EAST"]
            if dx == -1: return ["WEST"]
            if dy == 1: return ["SOUTH"]
            if dy == -1: return ["NORTH"]
        
        return ["PASS"]
    
    # ======================== ACCIONES DE TRABAJADORES ========================
    def _hand_action(self, hand_index: int) -> List[str]:
        """Acción para un trabajador contratado."""
        if hand_index >= len(self.hands_pos):
            return ["PASS"]
        x, y = self.hands_pos[hand_index]
        tile = self.tiles[y][x] if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]) else None
        
        # Similar a farmer pero con prioridad en cosechar y regar
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            if not tile.get("watered_today", True):
                return ["WATER"]
        
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            if tile.get("animal") and not tile.get("fed_today", True):
                if self.shed.get("WHEAT", 0) > 0:
                    return ["FEED"]
        
        # Si no hay nada, moverse a una parcela vacía o cercana
        return self._hand_move_to_useful(hand_index)
    
    def _hand_move_to_useful(self, hand_index: int) -> List[str]:
        """Movimiento para trabajador."""
        x, y = self.hands_pos[hand_index]
        # Buscar tile vacío o planta sin regar más cercano
        target = None
        min_dist = 100
        for ty in range(len(self.tiles)):
            for tx in range(len(self.tiles[ty])):
                tile = self.tiles[ty][tx]
                if tile is None or (isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", True)):
                    dist = abs(tx - x) + abs(ty - y)
                    if dist < min_dist:
                        min_dist = dist
                        target = (tx, ty)
        if target:
            dx = np.clip(target[0] - x, -1, 1)
            dy = np.clip(target[1] - y, -1, 1)
            if dx == 0 and dy == 0:
                return ["PASS"]
            if dx == 1: return ["EAST"]
            if dx == -1: return ["WEST"]
            if dy == 1: return ["SOUTH"]
            if dy == -1: return ["NORTH"]
        return ["PASS"]
    
    # ======================== ACCIONES DE MERCADO ========================
    def _market_actions(self) -> List[List[str]]:
        """Genera órdenes de mercado (compra/venta)."""
        orders = []
        money = self.money
        max_orders = self.env_cfg.get("maxMarketOrdersPerTurn", 10)
        
        # 1. Vender productos acumulados en el shed cuando el precio sea bueno
        for product, qty in list(self.shed.items()):
            if product in self.CROPS or product in ["FERTILIZER"] + list(self.ANIMALS.keys()):
                # Productos cosechados o fertilizante
                if self._is_price_good_to_sell(product):
                    # Vender una parte (no todo) para no colapsar el mercado
                    sell_qty = min(qty, max(1, qty // 3))  # vender 1/3
                    if sell_qty > 0:
                        orders.append(["SELL", product, sell_qty])
                        if len(orders) >= max_orders:
                            return orders[:max_orders]
        
        # 2. Comprar semillas según plan de siembra
        if self.phase != "late":  # al final no plantar más
            # Calcular cuántas semillas necesitamos
            empty_tiles = self._count_empty_tiles()
            seeds_needed = max(0, empty_tiles - self._count_seeds())
            if seeds_needed > 0:
                # Seleccionar mejores cultivos
                best_crops = self._select_best_crops(3)
                for crop in best_crops:
                    if self.seeds.get(crop, 0) < seeds_needed * 0.5:
                        cost = self.CROPS[crop]["seed_cost"]
                        if money >= cost:
                            orders.append(["BUY_SEED", crop, 1])
                            money -= cost
                            if len(orders) >= max_orders:
                                return orders[:max_orders]
        
        # 3. Comprar animales si tenemos estructuras
        if self.phase in ["early", "mid"]:
            # Verificar cuántas estructuras vacías tenemos
            empty_buildings = self._count_empty_buildings()
            if empty_buildings > 0:
                best_animals = self._select_best_animals()
                for animal in best_animals:
                    if self.shed.get(animal, 0) == 0 and money >= self.ANIMALS[animal]["cost"]:
                        orders.append(["BUY_ANIMAL", animal, 1])
                        money -= self.ANIMALS[animal]["cost"]
                        if len(orders) >= max_orders:
                            return orders[:max_orders]
        
        # 4. Comprar fertilizante si es barato y tenemos dinero
        if self.shed.get("FERTILIZER", 0) < 5 and money >= 100:
            fert_price = self._predict_price("FERTILIZER")
            if fert_price < 80:  # umbral
                orders.append(["BUY_PRODUCT", "FERTILIZER", 1])
                money -= fert_price
                if len(orders) >= max_orders:
                    return orders[:max_orders]
        
        # 5. Contratar trabajadores si necesitamos mano de obra
        workers = len(self.hands_pos)
        target = self._desired_workers()
        if workers < target and money >= self._hire_cost(workers):
            orders.append(["HIRE"])
            if len(orders) >= max_orders:
                return orders[:max_orders]
        
        # 6. Comprar terreno si tenemos suficiente dinero y estamos en fase media
        if self.phase in ["mid", "late"] and len(self.unlocked_quadrants) < 4:
            next_cost = self._land_cost()
            if money >= next_cost * 1.2:  # margen
                orders.append(["BUY_LAND"])
                if len(orders) >= max_orders:
                    return orders[:max_orders]
        
        return orders[:max_orders]
    
    def _count_empty_tiles(self) -> int:
        """Cuenta tiles vacíos o con malas hierbas (disponibles para plantar)."""
        count = 0
        for row in self.tiles:
            for tile in row:
                if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
                    count += 1
        return count
    
    def _count_seeds(self) -> int:
        return sum(self.seeds.values()) if self.seeds else 0
    
    def _count_empty_buildings(self) -> int:
        """Cuenta estructuras sin animal."""
        count = 0
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                    if tile.get("animal") is None:
                        count += 1
        return count
    
    def _desired_workers(self) -> int:
        """Número deseado de trabajadores según fase y tamaño de la granja."""
        if self.phase == "early":
            return min(1, len(self.unlocked_quadrants) + 1)
        elif self.phase == "mid":
            return min(3, len(self.unlocked_quadrants) + 2)
        else:  # late
            return min(5, len(self.unlocked_quadrants) + 3)
    
    def _hire_cost(self, current_workers: int) -> int:
        """Costo de contratar un nuevo trabajador."""
        # fib(n) donde n = número de contrataciones hoy
        n = self.hires_today + 1
        fib = [1, 1]
        for i in range(2, n+1):
            fib.append(fib[-1] + fib[-2])
        return self.env_cfg.get("farmHandCostMult", 1) * fib[n-1]
    
    def _land_cost(self) -> int:
        """Costo del siguiente terreno."""
        quadrants = len(self.unlocked_quadrants)
        costs = [1000, 2000, 4000]  # NW, NE, SW, SE (orden variable)
        if quadrants < len(costs):
            return costs[quadrants]
        return 10000  # no debería ocurrir
