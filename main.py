import ludopy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch import nn

from utilites import *

rng = np.random.default_rng()
flatten = lambda t: [item for sublist in t for item in sublist]

MOVE_OUT = 0
SAFE = 1
TRAVEL = 2
VULNERABLE = 3
ATTACKING = 4
HIT = 5
SUICIDE = 6
RUNWAY = 7
FINISH = 8

class Individual:
    def __init__(self, num_weights, bits_per_weight, crossover=False):
        # Size of genome depends on number of weights and bits per weight | num_weights * bits_per_weight
        self.num_weights = num_weights
        self.bits_per_weight = bits_per_weight
        self.fitness = 0

        self.genome = "" if crossover else self.random_genome(num_weights * bits_per_weight)

    @classmethod
    def crossover(cls, parent1, parent2):
        child1 = Individual(parent1.num_weights, parent1.bits_per_weight, crossover=True)
        child2 = Individual(parent1.num_weights, parent1.bits_per_weight, crossover=True)

        crossover_points = np.sort(rng.integers(0, len(parent1.genome), size=2))
        parent1_subgenome = parent1.genome[crossover_points[0]: crossover_points[1]]
        parent2_subgenome = parent2.genome[crossover_points[0]: crossover_points[1]]

        child1.genome = parent1.genome[:crossover_points[0]] + parent2_subgenome + parent1.genome[crossover_points[1]:]
        child2.genome = parent2.genome[:crossover_points[0]] + parent1_subgenome + parent2.genome[crossover_points[1]:]

        return child1, child2

    @staticmethod
    def random_genome(num_nucleotides):
        random_bits = [np.random.randint(2) for _ in range(num_nucleotides)]
        return ''.join(str(bit) for bit in random_bits)

    def mutate(self, mutation_rate):
        random_vals = rng.uniform(0, 1, size=(len(self.genome)))
        mutating_nucleotide_indeces = [i for i, v in enumerate(random_vals) if v < mutation_rate]
        for idx in mutating_nucleotide_indeces:
            mutated_bit = '0' if self.genome[idx] == '1' else '1'
            self.genome = self.genome[:idx] + mutated_bit + self.genome[idx+1:]

    def as_weights(self):
        genes = [self.genome[i:i + self.bits_per_weight] for i in
                 range(0, len(self.genome) - (self.bits_per_weight - 1), self.bits_per_weight)]
        return [(int(i, 2) - 2 ** (self.bits_per_weight-1)) / (2 ** self.bits_per_weight) for i in genes]

    def __str__(self):
        return self.genome


class Population:
    def __init__(self, population_size, num_weights, bits_per_weight):
        self.num_weights = num_weights
        self.bits_per_weight = bits_per_weight
        self.population_size = population_size

        self.members = []

    def spawn(self):
        for _ in range(self.population_size):
            self.members.append(Individual(self.num_weights, self.bits_per_weight))

    def total_fitness(self):
        total = 0
        for individual in self.members:
            total += individual.fitness
        return total

    def get_all_fitnesses(self):
        all_fitnesses = np.ndarray(self.population_size)
        for i, individual in enumerate(self.members):
            all_fitnesses[i] = individual.fitness
        return all_fitnesses

    def get_best_individual(self):
        best_fitness = -np.inf
        best_individual_idx = 0
        for i, individual in enumerate(self.members):
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_individual_idx = i
        return self.members[best_individual_idx]

    def __str__(self):
        pretty_pop = ""
        for i in self.members:
            pretty_pop += i.genome
            pretty_pop += '\n'
        return pretty_pop


class LudoGA:
    def __init__(self, use_ANN, population_size=100, bits_per_weight=10):
        # ANN Architecture
        self.use_ANN = use_ANN
        self.nodes_in_hidden_layers = [32]
        self.nodes_in_all_layers = [17, *self.nodes_in_hidden_layers, 4] if use_ANN else 9   # State space, hidden, action space | size of state transition function
        self.total_weights = calculate_total_weights(self.nodes_in_all_layers)
        self.model = self.setup_MLP() if use_ANN else None

        # Individual
        self.bits_per_weight = bits_per_weight
        self.best_individual = Individual(self.total_weights, self.bits_per_weight)

        # Population
        if population_size % 2 == 1:
            population_size += 1
            print("Population size cannot be an odd number. Rounded up to:", population_size)
        self.population_size = population_size   # Must be even due to how mating is performed
        self.population = Population(self.population_size, self.total_weights, self.bits_per_weight)
        self.population.spawn()

        # Evolution Params
        self.crossover_rate = 0.65
        self.mutation_rate = 0.001
        self.generation = 0
        self.max_generations = 0

        # Evolution Stats
        self.fitness_progression = []
        self.win_progression = []
        self.winrate_progression = []

        # Ludo specific
        self.num_games_to_play = 30
        self.player_idx = 0

        # For printing purposes
        self.chose_correctly = 0
        self.total_choices = 0
        self.choice_progression = []

    def setup_MLP(self):
        layers = []
        input_len = self.nodes_in_all_layers[0]
        output_len = self.nodes_in_all_layers[-1]
        for num_nodes in self.nodes_in_hidden_layers:
            layers.append(nn.Linear(input_len, num_nodes))
            layers.append(nn.ReLU())
            input_len = num_nodes

        layers.append(nn.Linear(input_len, output_len))
        layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*layers)

    def update_model_weights(self, individual: Individual):
        # map the genome of the individual to the weights of the NN architecture with correct dimensions
        weights = individual.as_weights()
        start_idx = end_idx = 0
        weight_matrix = []
        for i, num_nodes in enumerate(self.nodes_in_all_layers[:-1]):
            weight_matrix_between_layers = []
            for j in range(self.nodes_in_all_layers[i+1]):
                end_idx += num_nodes
                weight_matrix_between_layers.append(weights[start_idx:end_idx])
                start_idx = end_idx
                if j == self.nodes_in_all_layers[i+1] - 1:
                    end_idx += self.nodes_in_all_layers[i+1]
                    weight_matrix.append(weight_matrix_between_layers)
                    weight_matrix.append(weights[start_idx:end_idx])
                    start_idx = end_idx

        state_dict = self.model.state_dict()
        for i, name in enumerate(self.model.state_dict()):
            state_dict[name] = torch.tensor(weight_matrix[i])

    def update_choice_progression(self):
        self.choice_progression.append([self.chose_correctly, self.total_choices])
        self.chose_correctly = 0
        self.total_choices = 0

    def update_fitness(self):
        total_wins = 0
        fitnesses = []
        wins = []
        for individual in self.population.members:
            individual.fitness, num_wins = self.calculate_fitness(individual)
            total_wins += num_wins
            fitnesses.append(individual.fitness)
            wins.append(num_wins)

        if self.use_ANN:
            self.update_choice_progression()
        self.fitness_progression.append(fitnesses)
        self.win_progression.append(wins)
        winrate = 100 * total_wins / (self.num_games_to_play * self.population_size)
        print("Winrate: {:.2f}%".format(winrate))
        self.winrate_progression.append(winrate)

    def get_state(self, piece_pos, enemy_pieces, dice=0):
        piece_pos_next = piece_pos + dice
        if dice:
            if piece_pos == 0 and dice == 6:
                piece_pos_next = 1
            piece_pos = piece_pos_next

        state = np.zeros(9)
        home = [0]
        stars = [5, 12, 18, 25, 31, 38, 44, 51]
        home_globes = [1, 53]
        globes = [1, 9, 14, 22, 27, 35, 40, 48, 53]
        unsafe_globes = [14, 27, 40]

        distance_to_enemies = []
        for i, enemy in enumerate(enemy_pieces):
            for piece in enemy:
                dist = distance_between_two_pieces(piece_pos, piece, i + 1)
                distance_to_enemies.append(dist)

        vulnerable = any([-6 <= rel_pos < 0 for rel_pos in distance_to_enemies])
        attackable = any([0 < rel_pos <= 6 for rel_pos in distance_to_enemies])

        if piece_pos in globes:
            state[SAFE] = 1

        if (vulnerable and piece_pos not in globes) or piece_pos in unsafe_globes:
            state[VULNERABLE] = 1

        if attackable:
            state[ATTACKING] = 1

        if piece_pos >= 53:
            state[RUNWAY] = 1

        if piece_pos == 59:
            state[FINISH] = 1

        if dice:
            if piece_pos - 1 in home and dice == 6:
                state[MOVE_OUT] = 1

            if piece_pos_next in stars:
                state[TRAVEL] = 1

            if (piece_pos_next not in globes or piece_pos_next in home_globes) and 0 in distance_to_enemies and distance_to_enemies.count(0) < 2:
                state[HIT] = 1

            if 0 in distance_to_enemies and (distance_to_enemies.count(0) > 1 or (piece_pos_next in globes and piece_pos_next not in home_globes)):
                state[SUICIDE] = 1

        #print(piece_pos, dice, state)
        return state

    def determine_piece_to_move_ANN(self, player_pieces, enemy_pieces, dice, move_pieces):
        score = 0.1
        state = [*player_pieces, *flatten(enemy_pieces), dice]
        state_tensor = torch.Tensor(state)
        y = self.model(state_tensor)
        move = np.argmax(y.detach().numpy())
        if move in move_pieces:
            self.chose_correctly += 1
            piece_to_move = move
        else:
            score -= 0.11
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
        self.total_choices += 1
        return piece_to_move, score

    def determine_piece_to_move_state_transition(self, player_pieces_pos, enemy_pieces_pos, dice, move_pieces, solution: Individual):
        """
            Move out: when the piece can exit home and play the game
            Safe: when you enter a square where you can’t be hit home (globes or more than one piece on same square)
            Travel: when the piece can land on a star and travel to next star
            Vulnerable: 1-6 in front of enemy and not safe
            Attacking: 1-6 behind enemy and enemy is not safe
            Hit: when the piece can hit an enemy player home
            Suicide: when the piece can hit itself home
            Runway: Enter the 6 squares before goal
            Finish: piece has finished the game
        """
        possible_states = {}
        for move_piece in move_pieces:
            for piece_idx, piece_pos in enumerate(player_pieces_pos):
                possible_states[str(move_piece)] = possible_states.get(str(move_piece), np.zeros(9))\
                                                   + self.get_state(piece_pos, enemy_pieces_pos, dice=dice if move_piece == piece_idx else 0)
            #print()

        np_weights = np.asarray(solution.as_weights())

        best_move = move_pieces[0]
        best_score = -np.inf
        for move_piece in move_pieces:
            score = np.dot(possible_states[str(move_piece)], np_weights)
            if score > best_score:
                best_score = score
                best_move = move_piece

        return best_move, 0

    def determine_piece_to_move(self, player_pieces, enemy_pieces, dice, move_pieces, solution: Individual):
        if self.use_ANN:
            return self.determine_piece_to_move_ANN(player_pieces, enemy_pieces, dice, move_pieces)
        else:
            return self.determine_piece_to_move_state_transition(player_pieces, enemy_pieces, dice, move_pieces, solution)

    def calculate_fitness(self, solution: Individual):
        if self.use_ANN:
            self.update_model_weights(solution)
        fitness = 0
        num_wins = 0
        for i in range(self.num_games_to_play):
            g = ludopy.Game()
            player_is_a_winner = there_is_a_winner = False
            while not there_is_a_winner:
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                 there_is_a_winner), player_i = g.get_observation()
                if player_i == self.player_idx and len(move_pieces):
                    piece_to_move, score = self.determine_piece_to_move(player_pieces, enemy_pieces, dice, move_pieces, solution)
                    fitness += score
                else:
                    if len(move_pieces):
                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    else:
                        piece_to_move = -1

                _, _, _, _, player_is_a_winner, there_is_a_winner = g.answer_observation(piece_to_move)

            if player_is_a_winner and player_i == self.player_idx:
                num_wins += 1
                fitness += 10

        return fitness, num_wins

    def selection(self, ranked=False):
        best_of_generation = Population(self.population_size, self.total_weights, self.bits_per_weight)

        all_fitnesses = self.population.get_all_fitnesses()
        total_fitness = np.sum(all_fitnesses)
        all_fitnesses_normalized = all_fitnesses / total_fitness

        best_individual_in_population = self.population.get_best_individual()
        if best_individual_in_population.fitness >= self.best_individual.fitness:
            self.best_individual = best_individual_in_population
        print("Best Individual", max(all_fitnesses))
        print("Total Pop Fitness", total_fitness)

        if ranked:
            all_fitnesses_normalized = rank_from_normalised_values(all_fitnesses_normalized, normalise=True)
        accumulated_fitness = accumulate(all_fitnesses_normalized)

        for _ in range(self.population_size):
            roulette_val = rng.uniform(0, 1)
            index_of_selected_individual = index_of_first_element_above_value(accumulated_fitness, roulette_val)
            selected_individual = self.population.members[index_of_selected_individual]
            best_of_generation.members.append(selected_individual)
        return best_of_generation

    def reproduction(self, population: Population):
        new_generation = Population(self.population_size, self.total_weights, self.bits_per_weight)

        while population.members:
            index_of_random_parent1 = rng.integers(0, len(population.members))
            random_parent1 = population.members.pop(index_of_random_parent1)
            index_of_random_parent2 = rng.integers(0, len(population.members))
            random_parent2 = population.members.pop(index_of_random_parent2)

            crossover_val = rng.uniform(0, 1)
            if crossover_val < self.crossover_rate:
                child1, child2 = Individual.crossover(random_parent1, random_parent2)
            else:
                child1 = random_parent1
                child2 = random_parent2

            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)

            new_generation.members.append(child1)
            new_generation.members.append(child2)

        return new_generation

    def show_progression(self):
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_progression)
        plt.xlabel('Generations')
        plt.ylabel('Total Fitness')
        plt.xlim(0, self.max_generations)

        plt.subplot(1, 2, 2)
        plt.plot(self.win_progression)
        plt.xlabel('Generations')
        plt.ylabel('Win %')
        plt.ylim(0, 100)
        plt.xlim(0, self.max_generations)

        plt.tight_layout()
        plt.show()

    def evolve(self, max_generations, ranked):
        self.max_generations = max_generations
        pbar = tqdm(total=self.max_generations)
        while self.generation <= self.max_generations:
            self.generation += 1
            self.update_fitness()
            best_of_generation = self.selection(ranked=ranked)
            self.population = self.reproduction(best_of_generation)
            pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    ludoga = LudoGA(use_ANN=False, population_size=6, bits_per_weight=10)
    ludoga.evolve(max_generations=4, ranked=True)
    ludoga.show_progression()