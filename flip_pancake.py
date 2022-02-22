import DFBnB
import IDAStar
import random
import csv


def pancake_neighbours(state):
    for i in range(2, len(state) + 1):
        next_state = state[:i][::-1] + state[i:]
        yield (1, next_state, i)


def pancake_actions(state): # all flipping possible indices, no point to flip the top only
    return range(2, len(state) + 1)


def result(state, i):  #i is possible index of flip taken from actions
    return state[:i][::-1] + state[i:]


def pancake_h(s): #s is the state -  a tuple of length n
    return sum(abs(s[i] - s[i - 1]) > 1 for i in range(1, len(s)))


def pancake_goal(state):
    return tuple(sorted(state))


def generate_pancake_pile(n):
    test = list(range(1, n+1))
    random.shuffle(test)
    return tuple(test)


class pancake_test:
    def __init__(self,p, optimal_cost, cost, num_eval, algorithm, bound):
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

    is_goal = lambda p: p == pancake_goal(p)

    pancake_piles = []
    for i in range(25):
        pile = generate_pancake_pile(8)
        pancake_piles.append(pile)
    for i in range(25):
        pile = generate_pancake_pile(15)
        pancake_piles.append(pile)

    idastar_solver = IDAStar.IDAStar(pancake_h, pancake_neighbours)

    dfbnb_solver = DFBnB.DFBnB(pancake_h, pancake_neighbours)

    results = []

    for p in pancake_piles:
        print("start of IDA*")
        path, moves, cost, num_eval = idastar_solver.solve(p, is_goal, 80)
        optimal_cost = cost
        print("cost:", cost, "num_generated:", num_eval)
        pile_result = pancake_test(p, optimal_cost, cost, num_eval, 'IDA*', 0)
        results.append(pile_result.to_dict())
        print("start of DFBnB")
        ida_cost = cost
        for bound in range(70, 20, -10):
            path, moves, cost, num_eval = dfbnb_solver.solve(p, is_goal, bound)
            print("cost:", cost, "num_generated:", num_eval)
            pile_result = pancake_test(p,optimal_cost, cost, num_eval, 'DFBnB', bound)
            results.append(pile_result.to_dict())

    keys = results[0].keys()

    with open("pancake_results.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

