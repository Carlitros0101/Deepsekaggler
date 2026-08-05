import json
import os
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any

class Agent:
    def __init__(self, player: int, env_cfg: dict = None, config_path: str = "config.json"):
        self.player = player
        self.env_cfg = env_cfg or {}
        
        # Cargar configuración del agente
        self.config = self._load_config(config_path)
        
        # Parámetros del juego (hardcoded según documentación)
        self.CROPS = {
            "WHEAT": {"seed_cost": 10, "base_price": 25, "first_yield": 2, "max_yield": 6, "ongoing": False, "fert_boost": 2},
            "CARROT": {"seed_cost": 20, "base_price": 35, "first_yield": 2, "max_yield": 4, "ongoing": False, "fert_boost": 2},
            "TOMATO": {"seed_cost": 50, "base_price": 60, "first_yield": 8, "max_yield": 4, "ongoing": True, "fert_boost": 1},
            "STRAWBERRY": {"seed_cost": 100, "base_price": 120, "first_yield": 10, "max_yield": 4, "ongoing": True, "fert_boost": 1},
            "MELON": {"seed_cost": 80, "base_price": 250, "first_yield": 10, "max_yield": 6, "ongoing": False, "fert_boost": 2},
        }
        self.ANIMALS = {
            "GOOSE": {"cost": 300, "base_price": 50, "yield_interval": 1, "max_held": 4, "care_effect": 1},
            "COW": {"cost": 400, "base_price": 160, "yield_interval": 2, "max_held": 6, "care_effect": 1},
            "SHEEP": {"cost": 500, "base_price": 200, "yield_interval": 3, "max_held": 6, "care_effect": 1},
        }
        
        # Historiales para predicción
        self.price_history = defaultdict(lambda: deque(maxlen=20))
        self.inventory_history = defaultdict(lambda: deque(maxlen=20))
        self.shop_unlock_history = deque(maxlen=30)
        
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
        
        # Planificación y objetivos
        self.target_inventory = defaultdict(int)  # producto -> cantidad deseada en shed
        self.sell_schedule = {}  # producto -> (unidades_por_turno, turno_inicio)
        self.buy_schedule = {}   # producto -> (unidades_por_turno, turno_inicio)
        
        # Estadísticas de partidas anteriores (aprendizaje)
        self.stats = self._load_stats()
        self.current_game_stats = {
            "money_final": 0,
            "total_harvested": defaultdict(int),
            "total_sold": defaultdict(int),
            "total_bought": defaultdict(int),
        }
        
        # Parámetros ajustables (se actualizan con aprendizaje)
        self.params = self.config.get("params", {
            "sell_threshold": 1.1,      # vender si precio > media * threshold
            "buy_threshold": 0.85,      # comprar si precio < media * threshold
            "fert_price_limit": 80,     # precio máximo para comprar fertilizante
            "land_invest_phase": 0.6,   # fracción de dinero invertir en tierra
        })
        
    # ======================== CARGA DE CONFIGURACIÓN ========================
    def _load_config(self, path: str) -> dict:
        default_config = {
            "params": {
                "sell_threshold": 1.1,
                "buy_threshold": 0.85,
                "fert_price_limit": 80,
                "land_invest_phase": 0.6,
                "max_workers": 5,
                "inventory_targets": {
                    "WHEAT": 10,
                    "CARROT": 5,
                    "TOMATO": 3,
                    "STRAWBERRY": 2,
                    "MELON": 2,
                    "FERTILIZER": 5,
                }
            }
        }
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                return default_config
        return default_config
    
    def _save_config(self):
        with open("config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_stats(self) -> dict:
        if os.path.exists("stats.json"):
            try:
                with open("stats.json", 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_stats(self):
        # Guardar estadísticas acumuladas
        if not hasattr(self, 'stats_accum'):
            self.stats_accum = {}
        for k, v in self.current_game_stats.items():
            if k != "money_final":
                for subk, subv in v.items():
                    key = f"{k}_{subk}"
                    self.stats_accum[key] = self.stats_accum.get(key, 0) + subv
        with open("stats.json", 'w') as f:
            json.dump(self.stats_accum, f, indent=2)
    
    # ======================== ACTUACIÓN PRINCIPAL ========================
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._parse_obs(obs)
        self.day = obs["day"]
        self.step = obs.get("step", 0)
        
        # Actualizar historiales de mercado
        self._update_market_history(obs["market"])
        
        # Estimar demanda de la ciudad
        self._estimate_town_demand(obs["town"])
        
        # Calcular acciones
        farmer_actions = self._farmer_actions()
        hand_actions = [self._hand_action(i) for i in range(len(self.hands_pos))]
        market_actions = self._market_actions()
        
        # Guardar estadísticas de esta partida (al final)
        if self.day >= 29 and self.step % 10 == 0:
            self.current_game_stats["money_final"] = self.money
        
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
    
    def _update_market_history(self, market):
        for product, price in market["prices"].items():
            self.price_history[product].append(price)
        for product, inv in market["inventory"].items():
            self.inventory_history[product].append(inv)
    
    # ======================== PREDICCIÓN DE PRECIOS Y DEMANDA ========================
    def _predict_price(self, product: str, horizon: int = 1) -> float:
        """Predice precio usando media móvil + tendencia + estacionalidad."""
        hist = list(self.price_history[product])
        if len(hist) < 2:
            return self.CROPS.get(product, {}).get("base_price", 50)
        
        # Media móvil ponderada (más peso a lo reciente)
        weights = np.exp(np.linspace(-1, 0, len(hist)))
        weights /= weights.sum()
        avg = np.average(hist, weights=weights)
        
        # Tendencia
        if len(hist) >= 3:
            trend = (hist[-1] - hist[-3]) / 2
        else:
            trend = hist[-1] - hist[-2] if len(hist) >= 2 else 0
        
        # Estacionalidad diaria (si el producto es estacional, pero aquí no)
        pred = avg + trend * horizon
        return max(1, pred)
    
    def _estimate_town_demand(self, town):
        """Estima el consumo futuro de la ciudad."""
        shops = town.get("unlocked_shops", [])
        # Mapeo de tiendas a productos que consumen
        shop_demand = {
            "BAKERY": ["EGG", "WHEAT"],
            "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
            "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
            "YARN_STORE": ["WOOL"],
            "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
            "PET_CAFE": ["CARROT"],
            "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
            "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        }
        # Demanda esperada por producto (unidades por día)
        demand = defaultdict(int)
        for shop in shops:
            for product in shop_demand.get(shop, []):
                # Algunas tiendas consumen 2x (ej. YARN_STORE, PET_CAFE)
                multiplier = 2 if shop in ["YARN_STORE", "PET_CAFE"] else 1
                demand[product] += multiplier
        
        # Añadir consumo del centro de la ciudad
        # Consumo base: 1 de cada producto cada 12 turnos (≈2 por día)
        # A partir del día 10: 2 cada 12 turnos; día 20: 4 cada 12 turnos
        base_consumption = 1
        if self.day >= 20:
            base_consumption = 4
        elif self.day >= 10:
            base_consumption = 2
        for product in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]:
            demand[product] += base_consumption
        
        self.town_demand = demand
        return demand
    
    # ======================== OPTIMIZACIÓN DE INVENTARIO ========================
    def _get_inventory_target(self, product: str) -> int:
        """Devuelve la cantidad deseada de un producto en el shed."""
        targets = self.config["params"].get("inventory_targets", {})
        return targets.get(product, 3)
    
    def _inventory_gap(self, product: str) -> int:
        """Cantidad a comprar/vender para alcanzar el objetivo."""
        current = self.shed.get(product, 0)
        target = self._get_inventory_target(product)
        # Considerar también demanda de la ciudad
        future_demand = self.town_demand.get(product, 0) * 3  # 3 días de margen
        adjusted_target = target + future_demand
        return adjusted_target - current
    
    # ======================== ACCIONES DEL AGRICULTOR ========================
    def _farmer_actions(self) -> List[str]:
        x, y = self.farmer_pos
        tile = self.tiles[y][x] if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]) else None
        
        # 1. Cosechar plantas listas
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
        
        # 2. Cosechar animales / recoger fertilizante
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            if tile.get("fertilizer_available", False):
                return ["COLLECT_FERTILIZER"]
        
        # 3. Plantar si tenemos semillas y parcela vacía
        if tile is None and self.seeds:
            # Elegir cultivo más rentable según precios actuales
            best_crop = self._best_crop_to_plant()
            if best_crop and self.seeds.get(best_crop, 0) > 0:
                return ["PLANT", best_crop]
        
        # 4. Regar plantas sin agua
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not tile.get("watered_today", True):
                return ["WATER"]
        
        # 5. Alimentar / cuidar animales
        if tile and isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            animal = tile.get("animal")
            if animal:
                if not tile.get("fed_today", True):
                    if self.shed.get("WHEAT", 0) > 0:
                        return ["FEED"]
                if not tile.get("cared_today", True):
                    return ["CARE"]
        
        # 6. Fertilizar plantas si tenemos fertilizante
        if tile and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if self.shed.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < self.day:
                return ["FERTILIZE"]
        
        # 7. Colocar animales en estructuras vacías
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
        
        # 8. Construir estructuras si tenemos dinero y espacio
        if self._should_build_coop():
            return ["BUILD_COOP"]
        if self._should_build_pasture():
            return ["BUILD_PASTURE"]
        
        # 9. Moverse a un tile útil
        return self._move_to_useful()
    
    def _best_crop_to_plant(self) -> Optional[str]:
        """Selecciona el cultivo más rentable según precios actuales y demanda."""
        crops = list(self.CROPS.keys())
        scores = {}
        for crop in crops:
            price = self._predict_price(crop)
            cost = self.CROPS[crop]["seed_cost"]
            days = self.CROPS[crop]["first_yield"] + 2
            # Estimar yield
            if self.CROPS[crop]["ongoing"]:
                remaining = max(0, 30 - self.day - days)
                harvests = min(self.CROPS[crop]["max_yield"], remaining // 2 + 1)
                total_yield = harvests * 2
            else:
                total_yield = self.CROPS[crop]["max_yield"]
            # Ajustar por demanda de la ciudad
            demand_bonus = self.town_demand.get(crop, 0) * 0.1
            profit = (total_yield * price - cost) / days + demand_bonus
            scores[crop] = profit
        
        # Siempre dar prioridad al trigo si falta
        if self.shed.get("WHEAT", 0) < 5 and scores.get("WHEAT", 0) > 0:
            return "WHEAT"
        
        # Elegir el de mayor puntuación
        best = max(scores, key=scores.get) if scores else "WHEAT"
        return best
    
    def _should_build_coop(self) -> bool:
        """Decide si construir un gallinero."""
        # Si tenemos al menos un ganso en el shed y espacio
        if self.shed.get("GOOSE", 0) > 0:
            # Contar cuántos coops ya hay
            coop_count = sum(1 for row in self.tiles for t in row if isinstance(t, dict) and t.get("kind") == "COOP")
            if coop_count < 2 and self.money >= 0:  # BUILD_COOP es gratis
                return True
        return False
    
    def _should_build_pasture(self) -> bool:
        """Decide si construir un pastizal."""
        if self.shed.get("COW", 0) > 0 or self.shed.get("SHEEP", 0) > 0:
            pasture_count = sum(1 for row in self.tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")
            if pasture_count < 2:
                return True
        return False
    
    def _move_to_useful(self) -> List[str]:
        """Se mueve a un tile vacío o con trabajo pendiente."""
        x, y = self.farmer_pos
        target = None
        best_score = -1e9
        
        for ty in range(len(self.tiles)):
            for tx in range(len(self.tiles[ty])):
                tile = self.tiles[ty][tx]
                dist = abs(tx - x) + abs(ty - y)
                score = -dist
                if tile is None:
                    score += 5  # prefiere tiles vacíos para plantar
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    if not tile.get("watered_today", True) or tile.get("yield_units", 0) > 0:
                        score += 10
                elif isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                    if tile.get("animal") and (not tile.get("fed_today", True) or not tile.get("cared_today", True)):
                        score += 8
                    elif tile.get("fertilizer_available", False):
                        score += 6
                if score > best_score:
                    best_score = score
                    target = (tx, ty)
        
        if target and (target[0] != x or target[1] != y):
            dx = np.clip(target[0] - x, -1, 1)
            dy = np.clip(target[1] - y, -1, 1)
            if dx == 1: return ["EAST"]
            if dx == -1: return ["WEST"]
            if dy == 1: return ["SOUTH"]
            if dy == -1: return ["NORTH"]
        return ["PASS"]
    
    # ======================== ACCIONES DE TRABAJADORES ========================
    def _hand_action(self, hand_idx: int) -> List[str]:
        if hand_idx >= len(self.hands_pos):
            return ["PASS"]
        x, y = self.hands_pos[hand_idx]
        tile = self.tiles[y][x] if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]) else None
        
        # Prioridades: cosechar > regar > alimentar > plantar
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
        
        # Plantar si tenemos semillas y tile vacío
        if tile is None and self.seeds:
            best = self._best_crop_to_plant()
            if best and self.seeds.get(best, 0) > 0:
                return ["PLANT", best]
        
        # Moverse a un tile útil
        return self._hand_move_to_useful(hand_idx)
    
    def _hand_move_to_useful(self, hand_idx: int) -> List[str]:
        x, y = self.hands_pos[hand_idx]
        target = None
        best_score = -1e9
        for ty in range(len(self.tiles)):
            for tx in range(len(self.tiles[ty])):
                tile = self.tiles[ty][tx]
                dist = abs(tx - x) + abs(ty - y)
                score = -dist
                if tile is None:
                    score += 3
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    if not tile.get("watered_today", True) or tile.get("yield_units", 0) > 0:
                        score += 8
                elif isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                    if tile.get("animal") and not tile.get("fed_today", True):
                        score += 6
                if score > best_score:
                    best_score = score
                    target = (tx, ty)
        if target and (target[0] != x or target[1] != y):
            dx = np.clip(target[0] - x, -1, 1)
            dy = np.clip(target[1] - y, -1, 1)
            if dx == 1: return ["EAST"]
            if dx == -1: return ["WEST"]
            if dy == 1: return ["SOUTH"]
            if dy == -1: return ["NORTH"]
        return ["PASS"]
    
    # ======================== ACCIONES DE MERCADO ========================
    def _market_actions(self) -> List[List[str]]:
        orders = []
        max_orders = self.env_cfg.get("maxMarketOrdersPerTurn", 10)
        money = self.money
        
        # 1. Vender productos con excedente o buen precio
        for product, qty in list(self.shed.items()):
            if product in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]:
                # Ver si tenemos excedente
                target = self._get_inventory_target(product)
                gap = qty - target
                if gap > 0:
                    # Vender solo si precio es bueno
                    if self._is_price_good_to_sell(product):
                        sell_qty = min(gap, max(1, qty // 4))
                        orders.append(["SELL", product, sell_qty])
                        if len(orders) >= max_orders:
                            return orders[:max_orders]
                elif gap < 0 and self._is_price_good_to_buy(product):
                    # Comprar para alcanzar objetivo
                    buy_qty = min(-gap, 5)
                    if money >= self._predict_price(product) * buy_qty:
                        orders.append(["BUY_PRODUCT", product, buy_qty])
                        money -= self._predict_price(product) * buy_qty
                        if len(orders) >= max_orders:
                            return orders[:max_orders]
        
        # 2. Comprar semillas
        empty_tiles = self._count_empty_tiles()
        total_seeds = sum(self.seeds.values()) if self.seeds else 0
        if empty_tiles > total_seeds and self.day < 28:
            best_crops = self._best_crops_to_buy(3)
            for crop in best_crops:
                cost = self.CROPS[crop]["seed_cost"]
                if money >= cost and self.seeds.get(crop, 0) < 3:
                    orders.append(["BUY_SEED", crop, 1])
                    money -= cost
                    if len(orders) >= max_orders:
                        return orders[:max_orders]
        
        # 3. Comprar animales si tenemos estructuras vacías
        empty_buildings = self._count_empty_buildings()
        if empty_buildings > 0 and self.day < 25:
            for animal in self._best_animals_to_buy():
                if self.shed.get(animal, 0) == 0 and money >= self.ANIMALS[animal]["cost"]:
                    orders.append(["BUY_ANIMAL", animal, 1])
                    money -= self.ANIMALS[animal]["cost"]
                    if len(orders) >= max_orders:
                        return orders[:max_orders]
        
        # 4. Comprar fertilizante si barato y necesario
        fert_gap = self._inventory_gap("FERTILIZER")
        if fert_gap > 0:
            fert_price = self._predict_price("FERTILIZER")
            if fert_price < self.params.get("fert_price_limit", 80) and money >= fert_price:
                buy_qty = min(fert_gap, 5)
                orders.append(["BUY_PRODUCT", "FERTILIZER", buy_qty])
                money -= fert_price * buy_qty
                if len(orders) >= max_orders:
                    return orders[:max_orders]
        
        # 5. Contratar trabajadores
        workers = len(self.hands_pos)
        max_workers = self.params.get("max_workers", 5)
        if workers < max_workers and money >= self._hire_cost(workers):
            orders.append(["HIRE"])
            if len(orders) >= max_orders:
                return orders[:max_orders]
        
        # 6. Comprar terreno
        if len(self.unlocked_quadrants) < 4 and self.day > 5:
            next_cost = self._land_cost()
            invest_budget = money * self.params.get("land_invest_phase", 0.6)
            if invest_budget >= next_cost:
                orders.append(["BUY_LAND"])
                if len(orders) >= max_orders:
                    return orders[:max_orders]
        
        return orders[:max_orders]
    
    def _is_price_good_to_sell(self, product: str) -> bool:
        hist = list(self.price_history[product])
        if len(hist) < 3:
            return False
        avg = np.mean(hist[-5:])
        current = hist[-1]
        return current > avg * self.params.get("sell_threshold", 1.1)
    
    def _is_price_good_to_buy(self, product: str) -> bool:
        hist = list(self.price_history[product])
        if len(hist) < 3:
            return True  # comprar si no hay historial
        avg = np.mean(hist[-5:])
        current = hist[-1]
        return current < avg * self.params.get("buy_threshold", 0.85)
    
    def _best_crops_to_buy(self, n: int) -> List[str]:
        crops = list(self.CROPS.keys())
        scores = {}
        for crop in crops:
            price = self._predict_price(crop)
            cost = self.CROPS[crop]["seed_cost"]
            days = self.CROPS[crop]["first_yield"] + 2
            if self.CROPS[crop]["ongoing"]:
                remaining = max(0, 30 - self.day - days)
                harvests = min(self.CROPS[crop]["max_yield"], remaining // 2 + 1)
                total_yield = harvests * 2
            else:
                total_yield = self.CROPS[crop]["max_yield"]
            profit = (total_yield * price - cost) / days
            scores[crop] = profit
        # Asegurar trigo si falta
        if self.shed.get("WHEAT", 0) < 3:
            scores["WHEAT"] = scores.get("WHEAT", 0) + 10
        sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_crops[:n]]
    
    def _best_animals_to_buy(self) -> List[str]:
        animals = list(self.ANIMALS.keys())
        scores = {}
        for animal in animals:
            price = self._predict_price(animal.upper())
            cost = self.ANIMALS[animal]["cost"]
            interval = self.ANIMALS[animal]["yield_interval"]
            remaining = max(0, 30 - self.day - 2)
            harvests = remaining // interval
            total_yield = harvests * 2  # estimación
            wheat_price = self._predict_price("WHEAT")
            feed_cost = remaining * wheat_price * 0.5  # solo la mitad de los días necesita trigo?
            profit = (total_yield * price - cost - feed_cost) / max(1, remaining)
            scores[animal] = profit
        sorted_animals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_animals if a[1] > 0][:2]
    
    def _count_empty_tiles(self) -> int:
        count = 0
        for row in self.tiles:
            for tile in row:
                if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
                    count += 1
        return count
    
    def _count_empty_buildings(self) -> int:
        count = 0
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
                    if tile.get("animal") is None:
                        count += 1
        return count
    
    def _hire_cost(self, current_workers: int) -> int:
        n = self.hires_today + 1
        fib = [1, 1]
        for i in range(2, n+1):
            fib.append(fib[-1] + fib[-2])
        return self.env_cfg.get("farmHandCostMult", 1) * fib[n-1]
    
    def _land_cost(self) -> int:
        quadrants = len(self.unlocked_quadrants)
        costs = [1000, 2000, 4000]
        if quadrants < len(costs):
            return costs[quadrants]
        return 10000
    
    # ======================== APRENDIZAJE ========================
    def learn(self, final_money: int):
        """Actualiza parámetros basados en el resultado de la partida."""
        # Guardar estadísticas
        self.current_game_stats["money_final"] = final_money
        self._save_stats()
        
        # Ajustar umbrales basados en éxito
        if final_money > 5000:
            # Si ganamos mucho, aumentar umbral de venta (vender menos, esperar más)
            self.params["sell_threshold"] = min(1.3, self.params["sell_threshold"] + 0.02)
            self.params["buy_threshold"] = min(0.95, self.params["buy_threshold"] + 0.02)
        else:
            # Si perdemos, bajar umbrales (vender más barato, comprar más caro)
            self.params["sell_threshold"] = max(1.0, self.params["sell_threshold"] - 0.02)
            self.params["buy_threshold"] = max(0.7, self.params["buy_threshold"] - 0.02)
        
        # Guardar configuración actualizada
        self.config["params"] = self.params
        self._save_config()
