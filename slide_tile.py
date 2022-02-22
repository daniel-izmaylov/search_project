import DFBnB
import IDAStar
import random
import numpy as np
import csv


def slide_solved_state(n):
    return tuple(i % (n * n) for i in range(1, n * n + 1))


def slide_randomize(p, neighbours):
    for _ in range(len(p) ** 2):
        _, p, _ = random.choice(list(neighbours(p)))
    return p


def slide_neighbours(n):
    movelist = []
    for gap in range(n * n):
        x, y = gap % n, gap // n
        moves = []
        if x > 0: moves.append(-1)  # Move the gap left.
        if x < n - 1: moves.append(+1)  # Move the gap right.
        if y > 0: moves.append(-n)  # Move the gap up.
        if y < n - 1: moves.append(+n)  # Move the gap down.
        movelist.append(moves)

    def neighbours(p):
        gap = p.index(0)
        l = list(p)

        for m in movelist[gap]:
            l[gap] = l[gap + m]
            l[gap + m] = 0
            yield (1, tuple(l), (l[gap], m))
            l[gap + m] = l[gap]
            l[gap] = 0

    return neighbours


def slide_print(p):
    n = int(round(len(p) ** 0.5))
    l = len(str(n * n))
    for i in range(0, len(p), n):
        print(" ".join("{:>{}}".format(x, l) for x in p[i:i + n]))


def encode_cfg(cfg, n):
    r = 0
    b = n.bit_length()
    for i in range(len(cfg)):
        r |= cfg[i] << (b * i)
    return r


def gen_wd_table(n):
    goal = [[0] * i + [n] + [0] * (n - 1 - i) for i in range(n)]
    goal[-1][-1] = n - 1
    goal = tuple(sum(goal, []))

    table = {}
    to_visit = [(goal, 0, n - 1)]
    while to_visit:
        cfg, cost, e = to_visit.pop(0)
        enccfg = encode_cfg(cfg, n)
        if enccfg in table: continue
        table[enccfg] = cost

        for d in [-1, 1]:
            if 0 <= e + d < n:
                for c in range(n):
                    if cfg[n * (e + d) + c] > 0:
                        ncfg = list(cfg)
                        ncfg[n * (e + d) + c] -= 1
                        ncfg[n * e + c] += 1
                        to_visit.append((tuple(ncfg), cost + 1, e + d))

    return table


def slide_wd(n, goal):
    wd = gen_wd_table(n)
    goals = {i: goal.index(i) for i in goal}
    b = n.bit_length()

    def h(p):
        ht = 0  # Walking distance between rows.
        vt = 0  # Walking distance between columns.
        d = 0
        for i, c in enumerate(p):
            if c == 0: continue
            g = goals[c]
            xi, yi = i % n, i // n
            xg, yg = g % n, g // n
            ht += 1 << (b * (n * yi + yg))
            vt += 1 << (b * (n * xi + xg))

            if yg == yi:
                for k in range(i + 1, i - i % n + n):  # Until end of row.
                    if p[k] and goals[p[k]] // n == yi and goals[p[k]] < g:
                        d += 2

            if xg == xi:
                for k in range(i + n, n * n, n):  # Until end of column.
                    if p[k] and goals[p[k]] % n == xi and goals[p[k]] < g:
                        d += 2

        d += wd[ht] + wd[vt]

        return d

    return h


class puzzle_test:
    def __init__(self, p, optimal_cost, cost, num_eval, algorithm, bound):
        self.p = p
        self.cost = cost
        self.optimal_cost = optimal_cost
        self.num_eval = num_eval
        self.algorithm = algorithm
        self.bound = bound
        self.length = len(p)

    def to_dict(self):
        return {'start': self.p,
                'problem_size': self.length,
                'bound': self.bound,
                'optimal_cost': self.optimal_cost,
                'cost': self.cost,
                'states generated': self.num_eval,
                'algorithm': self.algorithm}


if __name__ == "__main__":
    results = []
    for n in [3, 4]:
        print(f'start {n**2-1} tile test')
        solved_state = slide_solved_state(n)
        neighbours = slide_neighbours(n)
        is_goal = lambda p: p == solved_state
        # problems = [tuple(np.random.permutation(16)) for i in range(1)]

        idastar_solver = IDAStar.IDAStar(slide_wd(n, solved_state), neighbours)

        dfbnb_solver = DFBnB.DFBnB(slide_wd(n, solved_state), neighbours)


        i = 0
        while i < 25:
            p = tuple(np.random.permutation(n**2))
            print(f'start puzzle {p}')
            try:
                path, moves, cost, num_eval = idastar_solver.solve(p, is_goal, 30)
                i += 1
            except:
                continue
            optimal_cost = cost
            puzzle_result = puzzle_test(p, optimal_cost, cost, num_eval, 'IDA*', 0)
            results.append(puzzle_result.to_dict())
            slide_print(p)
            print("cost:", cost, "num_generated:", num_eval)
            for bound in range(70, 30, -5):
                path, moves, cost, num_eval = dfbnb_solver.solve(p, is_goal, bound)
                print("cost:", cost, "num_generated:", num_eval)
                puzzle_result = puzzle_test(p, optimal_cost, cost, num_eval, 'DFBnB', bound)
                results.append(puzzle_result.to_dict())

    keys = results[0].keys()

    with open("n_tile_results.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

