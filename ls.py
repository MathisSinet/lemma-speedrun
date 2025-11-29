import math
import re
import sys

class BaseValue():
    def get_value(self, level: int) -> float:
        ...

class StepwiseValue(BaseValue):
    def __init__(self, offset = 0):
        self.offset = offset

    def get_value(self, level: int) -> float:
        cycles = level // 10
        mod = level % 10
        return 10 * (2 ** cycles - 1) + mod * 2 ** cycles + self.offset
    
class LinearValue(BaseValue):
    def __init__(self, power = 1, offset = 0):
        self.power = power
        self.offset = offset
    
    def get_value(self, level):
        return self.offset + self.power * level


class ExponentialValue(BaseValue):
    def __init__(self, base: float):
        self.base = base

    def get_value(self, level: int) -> float:
        return self.base ** level
    
class Cost():
    def __init__(self, base: float, scaling: float, firstFree: bool = False):
        self.base = base
        self.scaling = scaling
        self.firstFree = firstFree

    def get_cost(self, level: int):
        if self.firstFree:
            if level == 0:
                return 0
            else:
                return self.base * self.scaling ** (level - 1)
        else:
            return self.base * self.scaling ** level
        
class Variable():
    def __init__(self, cost: Cost, value: BaseValue, maxLevel: int | None = None, baseLevel: int = 0):
        self.cost_model = cost
        self.value_model = value
        self.level = baseLevel
        self.maxLevel = maxLevel
        self.baseLevel = baseLevel

    def value(self):
        return self.value_model.get_value(self.level)
    
    def cost(self):
        return self.cost_model.get_cost(self.level)
    
    def buy(self):
        self.level += 1
        return self.cost_model.get_cost(self.level - 1)
    
    def refund(self):
        self.level -= 1
        return self.cost_model.get_cost(self.level)

PACO_MAP = {
    "c1": "al",
    "c2": "bm",
    "c3": "cn",
    "c4": "do",
    "c5": "ep",
    "c6": "fq",
    "c7": "gr",
    "c8": "hs",
    "q1": "it",
    "q2": "ju"
}

REVERSE_PACO_MAP = {
    v[0]: k for k, v in PACO_MAP.items()
} | {
    v[0].upper(): k for k, v in PACO_MAP.items()
} | {
    v[1].upper(): k for k, v in PACO_MAP.items()
}

class RunParseError(ValueError):
    pass

class BaseLemma:
    def __init__(self):
        self.ticks = 0
        self.goal = 0.
        self.q = 0.
        self.variables : dict[str, Variable] = {}

        self.rho = 0.
        self.rho_total = 0.

        self.outputs: list[dict] = []
    

    def buy(self, var: str):
        self.rho -= self.variables[var].buy()


    def refund(self, var: str):
        self.rho += self.variables[var].refund()

    def get_rhodot(self, q: float) -> float:
        ...

    def tick(self):
        ...

    def get_best_distribution(self, vars: list[str], q: float, fund: float, mins: dict[str, int] | None = None) -> dict[str, int]:
        if mins is None:
            mins = dict()
        for var in vars:
            if var not in mins:
                mins[var] = self.variables[var].baseLevel

        best_rhodot = 0
        best_levels = {var: self.variables[var].baseLevel for var in vars}

        pivot = vars[0]
        self.variables[pivot].level = mins[pivot]
        fund -= sum(self.variables[pivot].cost_model.get_cost(level) for level in range(self.variables[pivot].baseLevel, mins[pivot]))

        while fund >= 0 and (not (self.variables[pivot].maxLevel and self.variables[pivot].level > self.variables[pivot].maxLevel)):
            if len(vars) > 1:
                cur_levels = self.get_best_distribution(vars[1:], q, fund, mins)
                for var in vars[1:]:
                    self.variables[var].level = cur_levels[var]
            else:
                cur_levels = dict()
            rhodot = self.get_rhodot(q)
            if rhodot > best_rhodot:
                best_rhodot = rhodot
                best_levels = {pivot: self.variables[pivot].level} | cur_levels
            fund -= self.variables[pivot].cost_model.get_cost(self.variables[pivot].level)
            self.variables[pivot].level += 1
            
        return best_levels


    def get_total_best_distribution(self, q: float):
        cur_levels = {var: self.variables[var].level for var in self.variables}
        for var in self.variables:
            self.variables[var].level = self.variables[var].baseLevel
        best_levels = self.get_best_distribution(list(self.variables.keys()), q, self.rho_total)
        for var in self.variables:
            self.variables[var].level = cur_levels[var]
        return best_levels


    def set_best_distribution(self, q: float):
        best_levels = self.get_total_best_distribution(q)
        for var in self.variables:
            while self.variables[var].level > best_levels[var]:
                self.refund(var)
            while self.variables[var].level < best_levels[var]:
                self.buy(var)


    def insert_output(self, q: float):
        self.outputs.append({
            "tick": self.ticks,
            "q": f"{q:.2f}",
            "rho": f"{self.rho:.2f}",
            "rho_total": f"{self.rho_total:.2f}"
        } | {
            var: self.variables[var].level for var in self.variables
        })


    def to_short_csv(self, var_order: list[str] | None = None):
        def extract(d: dict[str, int]):
            return {k: v for k, v in d.items() if k in self.variables}

        if var_order is None:
            var_order = sorted(extract(self.outputs[0]).keys())
        
        def get_items(line: dict):
            ret = [line["tick"], line["q"]]

            for var in var_order:
                if var in line:
                    ret.append(line[var])

            ret.append(line["rho"])
            ret.append(line["rho_total"])

            return ret
            

        for i in range(1, len(self.outputs) - 1):
            if extract(self.outputs[i-1]) == extract(self.outputs[i]) == extract(self.outputs[i+1]):
                continue
            print(*get_items(self.outputs[i]), sep=",")
        print(*get_items(self.outputs[-1]), sep=",")

    def to_pacowoc(self):
        last_op = 0
        paco_string = ""
        levels = [self.variables[var].baseLevel for var in self.variables]
        sorted_vars = sorted(self.variables.keys())

        for i, entry in enumerate(
                list(map(lambda d: {k: v for k, v in d.items() if k in self.variables}, self.outputs)) + 
                [{var: self.variables[var].baseLevel for var in self.variables}]
                ):
            for j, var in enumerate(sorted_vars):
                if entry[var] < levels[j]:
                    diff = levels[j] - entry[var]
                    levels[j] = entry[var]
                    if diff == 1:
                        paco_string += f"{PACO_MAP[var][1].upper()}{i - last_op}"
                    else:
                        paco_string += f"m{diff}{PACO_MAP[var][0]}{i - last_op}"
                    last_op = i
            for j, var in enumerate(sorted_vars):
                if entry[var] > levels[j]:
                    diff = entry[var] - levels[j]
                    levels[j] = entry[var]
                    if diff == 1:
                        paco_string += f"{PACO_MAP[var][0].upper()}{i - last_op}"
                    else:
                        paco_string += f"l{diff}{PACO_MAP[var][0]}{i - last_op}"
                    last_op = i
        paco_string += "K0"

        return paco_string
                            
    def parse_pacowoc_string(self, paco_string: str):
        self.outputs = [{var: self.variables[var].baseLevel for var in self.variables}]
        pos = 0

        #print(len(paco_string))

        paco_string = re.sub('n', 'l1', paco_string)
        paco_string = re.sub('o', 'm1', paco_string)

        parts = list(filter(None, re.split('([A-U]\\d+)|([lm]\\d+[a-k]\\d+)', paco_string, 0)))
        #print(len(''.join(parts)))


        for part in parts[:-1]:
            if m := re.match('([A-J])(\\d+)', part):
                varletter, delay = m.groups()
                var = REVERSE_PACO_MAP[varletter]
                if var not in self.variables:
                    raise RunParseError(f"Invalid variable {var}")
                for _ in range(int(delay)):
                    self.outputs.append(self.outputs[-1].copy())
                    pos += 1
                self.outputs[-1][var] += 1
                if self.variables[var].maxLevel and self.outputs[-1][var] > self.variables[var].maxLevel:
                    raise RunParseError(f"Variable level of {var} exceeded max level ({self.outputs[-1][var]}/{self.variables[var].maxLevel}) at position {pos}")

            elif m := re.match('([L-U])(\\d+)', part):
                varletter, delay = m.groups()
                var = REVERSE_PACO_MAP[varletter]
                if var not in self.variables:
                    raise RunParseError(f"Invalid variable {var}")
                for _ in range(int(delay)):
                    self.outputs.append(self.outputs[-1].copy())
                    pos += 1
                self.outputs[-1][var] -= 1
                if self.outputs[-1][var] < self.variables[var].baseLevel:
                    raise RunParseError(f"Variable level of {var} became lower than baseLevel ({self.outputs[-1][var]}/{self.variables[var].baseLevel}) at position {pos}")
                
            elif m := re.match('l(\\d+)([a-j])(\\d+)', part):
                bulk, varletter, delay = m.groups()
                var = REVERSE_PACO_MAP[varletter]
                if var not in self.variables:
                    raise RunParseError(f"Invalid variable {var}")
                for _ in range(int(delay)):
                    self.outputs.append(self.outputs[-1].copy())
                    pos += 1
                self.outputs[-1][var] += int(bulk)
                if self.outputs[-1][var] < self.variables[var].baseLevel:
                    raise RunParseError(f"Variable level of {var} became lower than baseLevel ({self.outputs[-1][var]}/{self.variables[var].baseLevel}) at position {pos}")

            elif m := re.match('m(\\d+)([a-j])(\\d+)', part):
                bulk, varletter, delay = m.groups()
                var = REVERSE_PACO_MAP[varletter]
                if var not in self.variables:
                    raise RunParseError(f"Invalid variable {var}")
                for _ in range(int(delay)):
                    self.outputs.append(self.outputs[-1].copy())
                    pos += 1
                self.outputs[-1][var] -= int(bulk)
                if self.outputs[-1][var] < 0:
                    raise RunParseError(f"Variable level of {var} became negative at position {pos}")

            else:
                raise RunParseError(f"Invalid part {part}")
        
        if m := re.match('K(\\d+)', parts[-1]):
            delay = m.group(1)
            for _ in range(int(delay)):
                self.outputs.append(self.outputs[-1].copy())
                pos += 1
        else:
            raise RunParseError(f"Invalid final part {part}")

        #print(self.outputs)

        return parts

    def run_from_pacowoc_string(self, string):
        self.parse_pacowoc_string(string)
        for levels in self.outputs[:-1]:
            for var in self.variables:
                while self.variables[var].level > levels[var]:
                    self.refund(var)
                    #print(f"On tick {self.ticks}, {var} was refunded to level {self.variables[var].level}")
            for var in self.variables:
                while self.variables[var].level < levels[var]:
                    if self.rho < self.variables[var].cost() and self.variables[var].cost() != 0:
                        print("Error on tick", self.ticks)
                        raise RunParseError(f"Unable to purchase {var} level {self.variables[var].level + 1} : rho {self.rho} is less than the cost {self.variables[var].cost()}")
                    self.buy(var)
                    #print(f"On tick {self.ticks}, {var} was bought to level {self.variables[var].level}")
            
            self.tick()
            #print(f"On tick {self.ticks}, rho are {self.rho:.2f}, {self.rho_total:.2f}")
            self.ticks += 1
        
        for var in self.variables:
            while self.variables[var].level > self.outputs[-1][var]:
                self.refund(var)
            while self.variables[var].level < self.outputs[-1][var]:
                self.buy(var)
        if self.rho < self.goal:
            raise RunParseError(f"Insufficient rho to finish the run: {self.rho:.2e}, {self.rho_total:.2e}")
        
        print("Run successful")
        print((len(self.outputs) - 1) / 10)


class Lemma1(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e10
        self.variables = {
            "c1" : Variable(
                Cost(10, 1.5, True),
                StepwiseValue()
            ),
            "c2" : Variable(
                Cost(30, 3),
                ExponentialValue(2)
            ),
            "c3" : Variable(
                Cost(100, 3),
                ExponentialValue(2)
            )
        }

    def get_rhodot(self, q: float, dt: float = 0.1):
        qt0 = q
        qt1 = q + dt

        c1 = self.variables["c1"].value()
        c2 = self.variables["c2"].value()
        c3 = self.variables["c3"].value()

        return c1 * (dt * (1/2 * c2 + c3) + c2 * (math.cos(qt0) - math.cos(qt1)))
    
    def get_total_best_distribution(self, q: float):
        current_levels = {var: self.variables[var].level for var in self.variables}

        for var in self.variables:
            self.variables[var].level = 0
        
        best_dist = self.get_best_distribution(
            list(self.variables.keys()), 
            q, 
            self.rho_total, 
            {"c3":
                0
                #current_levels["c3"]
            })

        for var in self.variables:
            self.variables[var].level = current_levels[var]

        return best_dist

    def tick(self):
        # Interpolation bs
        self.rho += self.get_rhodot(self.q)
        self.rho_total += self.get_rhodot(self.q)
        self.q += 0.1

    def run(self):
        while self.rho_total < self.goal:
            self.set_best_distribution(self.q)
            self.tick()
            self.insert_output(self.q - 0.1)
            self.ticks += 1

        self.to_short_csv()

        print(self.to_pacowoc())

class Lemma2(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e8
        self.q = 1.
        self.variables = {
            "c1" : Variable(
                Cost(10, 1.5, True),
                StepwiseValue()
            ),
            "c2" : Variable(
                Cost(30, 3),
                ExponentialValue(2)
            ),
            "c3" : Variable(
                Cost(200, 1.3),
                StepwiseValue()
            ),
            "c4": Variable(
                Cost(250, 2),
                ExponentialValue(2)
            )
        }

    def get_rhodot(self, q: float):
        #print([(var, self.variables[var].level) for var in self.variables], q)
        return (self.variables["c1"].value() * self.variables["c2"].value() + self.variables["c3"].value() * self.variables["c4"].value()) \
            / q ** (sum(variable.level for variable in self.variables.values()) / 100)

    def get_total_best_distribution(self, q: float):
        current_levels = {var: self.variables[var].level for var in self.variables}

        for var in self.variables:
            self.variables[var].level = 0
        
        best_c12_dist = self.get_best_distribution(["c1", "c2"], q, self.rho_total) | {"c3": 0, "c4": 0}
        for var in best_c12_dist:
            self.variables[var].level = best_c12_dist[var]
        c12_rhodot = self.get_rhodot(q)

        for var in self.variables:
            self.variables[var].level = 0
        
        best_c34_dist = self.get_best_distribution(["c3", "c4"], q, self.rho_total) | {"c1": 0, "c2": 0}
        for var in best_c34_dist:
            self.variables[var].level = best_c34_dist[var]
        c34_rhodot = self.get_rhodot(q)

        for var in self.variables:
            self.variables[var].level = current_levels[var]

        if c12_rhodot >= c34_rhodot:
            return best_c12_dist
        else:
            return best_c34_dist

    def tick(self):
        self.q += 0.1
        self.rho += self.get_rhodot(self.q) * 0.1
        self.rho_total += self.get_rhodot(self.q) * 0.1

    def run(self):
        while self.rho_total < self.goal:
            self.set_best_distribution(self.q + 0.1)
            self.tick()
            self.insert_output(self.q - 0.1)
            self.ticks += 1

        #self.to_short_csv()
        print(self.to_pacowoc())

class Lemma3(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e20
        self.variables = {
            "q1": Variable(
                Cost(10, 4, True),
                StepwiseValue()
            ),
            "q2": Variable(
                Cost(50, 50),
                ExponentialValue(2)
            ),
            "c1": Variable(
                Cost(1e4, 3),
                LinearValue()
            ),
            "c2": Variable(
                Cost(1e5, 2),
                ExponentialValue(2),
                maxLevel = 25
            ),
            "c3": Variable(
                Cost(100, 100),
                ExponentialValue(2)
            )
        }
    
    def get_rhodot(self, q: float, dt: float = 0.1):
        qt0 = q
        qt1 = q + dt * self.get_qdot()

        c1 = self.variables["c1"].value()
        c2 = self.variables["c2"].value()
        c3 = self.variables["c3"].value()

        return dt * ((-2) ** c1 * c2 + c3 * (qt0 + qt1) / 2)

    def get_qdot(self):
        return self.variables["q1"].value() * self.variables["q2"].value()
    
    def tick(self):
        # q interpolation
        self.rho += self.get_rhodot(self.q)
        self.rho_total += self.get_rhodot(self.q)
        self.q += self.get_qdot() * 0.1

class Lemma4_C3Value(BaseValue):
    def get_value(self, level):
        return float((level + 1) ** 2)

class Lemma4(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e10
        self.variables = {
            "c1" : Variable(
                Cost(1, 2.87, True),
                StepwiseValue()
            ),
            "c2" : Variable(
                Cost(5000, 10),
                ExponentialValue(2)
            ),
            "c3" : Variable(
                Cost(1, 10),
                Lemma4_C3Value()
            )
        }

    def get_rhodot(self, q: float):
        return (
            self.variables["c1"].value() *
            self.variables["c2"].value() *
            (self.variables["c3"].value() * q - (q ** 2) / 5)
        )
    
    def get_total_best_distribution(self, q: float):
        current_levels = {var: self.variables[var].level for var in self.variables}

        for var in self.variables:
            self.variables[var].level = 0
        
        best_dist = self.get_best_distribution(
            list(self.variables.keys()), 
            q, 
            self.rho_total, 
            {"c3":
                0
                #current_levels["c3"]
            })

        for var in self.variables:
            self.variables[var].level = current_levels[var]

        return best_dist

    def tick(self):
        self.q += 0.1
        self.rho += self.get_rhodot(self.q) * 0.1
        self.rho_total += self.get_rhodot(self.q) * 0.1

    def run(self):
        while self.rho_total < self.goal:
            self.set_best_distribution(self.q + 0.1)
            self.tick()
            self.insert_output(self.q - 0.1)
            self.ticks += 1

        self.to_short_csv()

        #print(self.to_pacowoc())

class Lemma5(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e25
        self.variables = {
            "q1": Variable(
                Cost(10, 3),
                StepwiseValue(1)
            ),
            "q2": Variable(
                Cost(30, 10),
                ExponentialValue(2)
            ),
            "c1": Variable(
                Cost(0, 0),
                LinearValue()
            ),
            "c2": Variable(
                Cost(1e6, 1.1),
                LinearValue()
            ),
            "c3": Variable(
                Cost(1e11, 1.1),
                LinearValue()
            ),
            "c4": Variable(
                Cost(1e13, 1.1),
                LinearValue()
            ),
            "c5": Variable(
                Cost(1e15, 1.08),
                LinearValue()
            ),
            "c6": Variable(
                Cost(1e17, 1.06),
                LinearValue()
            ),
            "c7": Variable(
                Cost(1e19, 1.02),
                LinearValue()
            ),
            "c8": Variable(
                Cost(1e21, 1.01),
                LinearValue()
            )
        }

    def get_rhodot(self, q: float):
        return sum(self.variables[f"c{i}"].value()**4 * (2 * i**2 - self.variables[f"c{i}"].value()) * q for i in range(1, 9))
    
    def get_qdot(self):
        return self.variables["q1"].value() * self.variables["q2"].value()

    def tick(self):
        self.q += self.get_qdot() * 0.1
        self.rho += self.get_rhodot(self.q) * 0.1
        self.rho_total += self.get_rhodot(self.q) * 0.1

class Lemma6Value(BaseValue):
    def __init__(self, power):
        self.power = power
    
    def get_value(self, level):
        return level ** self.power

class Lemma6(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e15
        self.variables = {
            "q1": Variable(
                Cost(10, 5),
                StepwiseValue(1)
            ),
            "q2": Variable(
                Cost(100, 10),
                ExponentialValue(2)
            ),
            "c1": Variable(
                Cost(30, 10, True),
                StepwiseValue(1)
            ),
            "c2": Variable(
                Cost(30, 10, True),
                StepwiseValue(1)
            ),
            "c3": Variable(
                Cost(1e6, 1.15),
                Lemma6Value(1 / math.e),
                baseLevel = 2
            ),
            "c4": Variable(
                Cost(1e6, 1.15),
                Lemma6Value(1 / math.pi)
            )
        }

    def get_rhodot(self, q: float, dt: float = 0.1):
        qt0 = q
        qt1 = q + dt * self.get_qdot()

        c1 = self.variables["c1"].value()
        c2 = self.variables["c2"].value()
        c3 = self.variables["c3"].value()
        c4 = self.variables["c4"].value()

        return dt * (c1 - c2) / (c3 - c4) * (qt0 + qt1) / 2
    
    def get_qdot(self):
        return self.variables["q1"].value() * self.variables["q2"].value()
    
    def tick(self):
        # q interpolation
        self.rho += self.get_rhodot(self.q)
        self.rho_total += self.get_rhodot(self.q)
        self.q += self.get_qdot() * 0.1

    def get_all_c34(self):
        self.variables["c3"].level = 2
        self.variables["c4"].level = 0
        best_approx = self.variables["c3"].value() - self.variables["c4"].value()
        approx_list = []
        for c4_lvl in range(0, 100):
            for c3_lvl in range(2, 100):
                self.variables["c3"].level, self.variables["c4"].level = c3_lvl, c4_lvl
                cost = (sum(self.variables["c3"].cost_model.get_cost(i) for i in range(2, c3_lvl)) + 
                        sum(self.variables["c4"].cost_model.get_cost(i) for i in range(c4_lvl)))
                approx = self.variables["c3"].value() - self.variables["c4"].value()
                if abs(approx) <= abs(best_approx):
                    best_approx = approx
                    approx_list.append([c3_lvl, c4_lvl, approx, cost])
        approx_list.sort(key = lambda l: abs(l[2]), reverse = True)
                
        for line in approx_list:
            print(*line, sep=",")



class Lemma7(BaseLemma):
    def __init__(self):
        super().__init__()

        self.goal = 1e15
        self.variables = {
            "q1": Variable(
                Cost(10, 1.5, True),
                StepwiseValue()
            ),
            "q2": Variable(
                Cost(30, 10),
                ExponentialValue(2)
            ),
            "c1": Variable(
                Cost(10000, 1.2),
                LinearValue(1, 1)
            ),
            "c2": Variable(
                Cost(10000, 1.5),
                LinearValue(1, 1)
            )
        }
    
    def get_rhodot(self, q: float, dt: float = 0.1):
        qt0 = q
        qt1 = q + dt * self.get_qdot()

        c1 = self.variables["c1"].value()
        c2 = self.variables["c2"].value()

        return dt * (qt0 + qt1) / 2 / abs(math.e - c1 / c2)

    def get_qdot(self):
        return self.variables["q1"].value() * self.variables["q2"].value()
    
    def tick(self):
        # q interpolation
        self.rho += self.get_rhodot(self.q)
        self.rho_total += self.get_rhodot(self.q)
        self.q += self.get_qdot() * 0.1

    def get_all_c12(self):
        self.variables["c1"].level = 0
        self.variables["c2"].level = 0
        best_approx = math.e - self.variables["c1"].value() / self.variables["c2"].value()
        approx_list = []
        for c2_lvl in range(0, 150):
            for c1_lvl in range(0, 300):
                self.variables["c1"].level, self.variables["c2"].level = c1_lvl, c2_lvl
                cost = (sum(self.variables["c1"].cost_model.get_cost(i) for i in range(c1_lvl)) + 
                        sum(self.variables["c2"].cost_model.get_cost(i) for i in range(c2_lvl)))
                approx = math.e - self.variables["c1"].value() / self.variables["c2"].value()
                if abs(approx) < abs(best_approx):
                    best_approx = approx
                    approx_list.append([c1_lvl, c2_lvl, approx, cost])
        approx_list.sort(key = lambda l: abs(l[2]), reverse = True)
                
        for line in approx_list:
            print(*line, sep=",")


def verify_lemma():
    lemma_id = int(input("Lemma id: "))
    if lemma_id not in range(1, 8):
        raise ValueError("Invalid Lemma id")
    LEMMAS: list[BaseLemma] = [
        BaseLemma(),
        Lemma1(),
        Lemma2(),
        Lemma3(),
        Lemma4(),
        Lemma5(),
        Lemma6(),
        Lemma7()
    ]

    run = input()

    LEMMAS[lemma_id].run_from_pacowoc_string(run)

verify_lemma()
