import numpy as np
from matplotlib import pyplot as plt

def accumulate(list):
    for i, el in enumerate(list):
        if i != 0:
            list[i] += list[i-1]
    return list


def rank_from_normalised_values(list, normalise=False):
    rank = 1
    for i, el in enumerate(list):
        min_idx = np.argmin(list)
        list[min_idx] = rank
        rank += 1
    return list / np.sum(list) if normalise else list


def index_of_first_element_above_value(list, val):
    index = 0
    for i, el in enumerate(list):
        if el > val:
            index = i
            break
    return index


def calculate_total_weights(nodes_in_layers):
    total_weights = 0
    if type(nodes_in_layers) == int:
        return nodes_in_layers
    if len(nodes_in_layers) == 1:
        total_weights = nodes_in_layers[0]
    for i in range(len(nodes_in_layers)-1):
        weights_between_layers = nodes_in_layers[i] * nodes_in_layers[i+1]
        biases = nodes_in_layers[i + 1]
        total_weights += weights_between_layers + biases
    return total_weights


def distance_between_two_pieces(piece, enemy, i):
    if enemy == 0 or enemy >= 53 or piece == 0 or piece >= 53:
        return 1000
    enemy_relative_to_piece = (enemy + 13 * i) % 52
    if enemy_relative_to_piece == 0: enemy_relative_to_piece = 52
    distances = [enemy_relative_to_piece - piece, (enemy_relative_to_piece - 52) - piece]
    return distances[np.argmin(list(map(abs, distances)))]


def write_list_to_file(list, file_name):
    with open(file_name + '.txt', 'a') as f:
        f.write('[')
        for item in list:
            f.write(str(item))
            f.write(', ')
        f.write(']')

def write_matrix_to_file(matrix, file_name):
    with open(file_name + '.txt', 'a') as f:
        for list in matrix:
            f.write('[')
            for item in list:
                f.write(str(item))
                f.write(', ')
            f.write(']\n')

def show_choice(filepath):
    choice_rate = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            row = line[1:-4].split(', ')
            choice_rate.append(100 * (int(row[0]) / int(row[1])))

    plt.plot(choice_rate)
    print(np.mean(choice_rate))
    plt.xlabel('Generations')
    plt.ylabel('Valid choice %')
    plt.ylim(0, 100)
    plt.yticks(ticks=np.arange(0, 101, 10), labels=[str(item) for item in np.arange(0, 101, 10)])
    plt.xlim(0, 100)
    plt.show()

def show_box(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            print(line[1:-4].split(', '))
            vals = [100 * float(item) / 30 for item in line[1:-4].split(', ')]
            matrix.append(vals)

    plt.boxplot(matrix)
    plt.xlabel('Generations')
    plt.ylabel('Win %')
    plt.yticks(ticks=np.arange(0, 101, 10), labels=[str(item) for item in np.arange(0, 101, 10)])
    plt.xticks(ticks=np.arange(0, 101, 20), labels=[str(item) for item in np.arange(0, 101, 20)])
    plt.xlim(0, 100)
    plt.show()


def show_data(filepath):
    with open(filepath, 'r') as f:
        winrates = f.readline()[1:-3].split(', ')
        winrates = [float(num) for num in winrates]
    plt.xlabel('Generations')
    plt.ylabel('Win %')
    plt.yticks(np.arange(0,101, 5))
    plt.xticks(np.arange(0, 101, 20))
    plt.ylim(20, 50)
    plt.xlim(0, 100)
    plt.plot(winrates)
    print(np.mean(winrates))
    plt.text(winrates.index(max(winrates)), max(winrates)+1, str('%.2f' % max(winrates)), horizontalalignment='center', verticalalignment='center')
    plt.text(15, np.mean(winrates)+1.1, str('%.2f' % np.mean(winrates)), horizontalalignment='center', verticalalignment='center')
    plt.axhline(y=max(winrates), color='y', linestyle=':')
    plt.axhline(y=np.mean(winrates), color='g', linestyle=':')
    plt.show()