
# difficulty levels for certain Matrix size based on the examples in the website: https://www.futoshiki.org/
import sys
import random as rand
import numpy as np
import copy
from statistics import mean

import matplotlib.pyplot as plt  # for algorithm analysis plot


GENERATION = 0  # will be updated
MATRIX = []  # will save the current matrix

SIZE = 5  # generic size 5*5
LIMIT = 2000  # maximum number of generations
POPULATION_SIZE = 100  # number of matrix in each generation
INITIALS = {}  # all constant numbers given at the beginning
INEQUALITIES = {}  # (x1, y1) > SYMBOLS[(x1, y1)] = (x2, y2)
TOTAL_RESTRICTIONS = SIZE + SIZE  # minimum number of restrictions (5 + 5 matrix)
CURRENT_POPULATION = []

FIRST_MIN = TOTAL_RESTRICTIONS
BEST_MIN = FIRST_MIN
BEST_MAT = []  # the final best matrix (this is a matrix!)
MIN_LIST = []  # a list of the best fitness score in each generation (this is a list!)
AVG_LIST = []  # a list of average score in each generation


def update_matrix():
    global GENERATION, MATRIX, SIZE, INITIALS
    if GENERATION == 0:
        initialize_matrix()
    if GENERATION == 1:
        for i in range(SIZE):  # [0:4]
            for j in range(SIZE):  # [0:4]
                if MATRIX[i][j] == 0 or initial_value(i, j) == 0:
                    MATRIX[i][j] = rand.randint(1, SIZE)
    else:
        pass


def initialize_matrix():
    global MATRIX
    for i in range(SIZE):  # [0..SIZE-1]
        row = []
        for j in range(SIZE):  # [0..SIZE-1]
            if initial_value(i, j) == 0:
                row.append("□")
            else:
                row.append(initial_value(i, j))
        MATRIX.append(row)


def print_mat(matrix):
    if GENERATION > -1:
        for line in matrix:
            for item in line:
                print(item, end="  ")
            print()


def fix_mat(matrix):  # insure that initial values are in the matrix
    for i in range(SIZE):  # [0..SIZE-1]
        for j in range(SIZE):  # [0..SIZE-1]
            if initial_value(i, j) != 0:
                matrix[i][j] = initial_value(i, j)
    return matrix


def initial_value(i, j):
    x = i + 1
    y = j + 1
    val = 0
    if (x, y) in INITIALS.keys():  # this location is immutable
        val = INITIALS[(x, y)]  # the value of the key
    return val  # 0 or num


def get_matrix_size():  # from user
    while True:
        size = input("Matrix size (5, 6 or 7): ")
        if size == "5" or size == "6" or size == "7":
            break
        else:
            print("Wrong input, please try again")
    return int(size)


def get_difficulty():  # from user
    while True:
        difficulty = input("Difficulty level (easy or tricky): ")
        if difficulty == "easy" or difficulty == "tricky":
            break
        else:
            print("Wrong input, please try again")
    return difficulty


def get_algorithm():  # from user
    while True:
        algorithm = input("\nPlease choose algorithm (Normal, Darwin or Lamarck): ")
        if algorithm == "Normal" or algorithm == "Darwin" or algorithm == "Lamarck":
            break
        else:
            print("Wrong input, please try again")
    return algorithm


def mat_score(matrix):
    score = TOTAL_RESTRICTIONS
    for row in matrix:  # check matrix rows
        if len(set(row)) == len(row):  # All elements in the row are unique
            score -= 1
    for col in np.transpose(matrix):  # transpose matrix and check matrix columns
        if len(set(col)) == len(col):  # All elements in the col are unique
            score -= 1

    for coordinate in INEQUALITIES:  # all (x1, y1) >
        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = INEQUALITIES[coordinate][0]
        y2 = INEQUALITIES[coordinate][1]
        if matrix[x1-1][y1-1] > matrix[x2-1][y2-1]:  # check if '>' is true in the specific locations
            score -= 1

    return score  # we aspire for the lowest score possible!


def transpose_mat(matrix):
    trans = []
    for i in range(SIZE):  # [0..SIZE-1]
        row = []
        for j in range(SIZE):  # [0..SIZE-1]
            row.append(matrix[j][i])
        trans.append(row)
    return trans


def optimize(matrix, current_score):  # optimize current matrix by checking each row to see if a number is missing
    counter = current_score
    old_mat = matrix
    old_score = mat_score(old_mat)
    new_mat = matrix

    for i in range(SIZE):  # [0..SIZE-1]
        row = new_mat[i]
        if len(set(row)) != len(row):  # at least two numbers in the row are the same
            for j in range(SIZE):  # [0..SIZE-1]  check the numbers in current row
                num = j + 1  # 1,2,3,4..SIZE
                if row.count(num) == 0 and initial_value(i, j) == 0 and row.count(new_mat[i][j]) == 2:
                    row[j] = num
                    counter -= 1

    if counter > 0:  # in this case we transpose and check "columns"
        new_mat = transpose_mat(new_mat)
        for i in range(SIZE):  # [0..SIZE-1]
            row = new_mat[i]
            if len(set(row)) != len(row):  # at least two numbers in the row are the same
                for j in range(SIZE):  # [0..SIZE-1]  check the numbers in current row
                    num = j + 1  # 1,2,3,4..SIZE
                    if row.count(num) == 0 and initial_value(i, j) == 0 and row.count(new_mat[i][j]) == 2:
                        row[j] = num
                        counter -= 1
                        if counter == 0:
                            break

    new_mat = transpose_mat(new_mat)  # transpose back to the original matrix

    if counter > 0:  # check INEQUALITIES conditions
        for key in INEQUALITIES.keys():
            x1 = key[0]
            y1 = key[1]
            x2 = INEQUALITIES[key][0]
            y2 = INEQUALITIES[key][1]

            if matrix[x1-1][y1-1] < matrix[x2-1][y2-1]:
                temp = matrix[x2-1][y2-1]
                matrix[x2 - 1][y2 - 1] = matrix[x1-1][y1-1]
                matrix[x1 - 1][y1 - 1] = temp
                counter -= 1
                if counter == 0:
                    break

    new_score = mat_score(new_mat)

    if new_score <= old_score:
        return fix_mat(new_mat)  # "improved" matrix
    else:
        return fix_mat(old_mat)


# convert the two dimensional matrix list to one dimensional list
def mat_to_list(mat):
    one_dim_list = []
    for row in mat:
        for sqr in row:
            one_dim_list.append(sqr)
    return one_dim_list


# convert a one dimensional list to two dimensional matrix list
def list_to_mat(one_dim_list):
    mat = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(one_dim_list[SIZE * i + j])
        mat.append(row)
    return fix_mat(mat)


def crossover(mat1, mat2):
    list1, list2 = mat_to_list(mat1), mat_to_list(mat2)
    ret_bin = []
    # generate a random index
    cut = rand.randint(1, SIZE ** 2 - 2)
    # the random index is dividing the one dimensional matrix list into 2 parts
    # the new one dimensional matrix list of numbers is part from the first list and part from the second list
    for i in range(cut):
        ret_bin.append(list1[i])
    for i in range(cut, SIZE ** 2):
        ret_bin.append(list2[i])
    ret_mat = list_to_mat(ret_bin)
    return fix_mat(ret_mat)


# mutation function - return new matrix after mutation
def mutation(matrix):
    binary = mat_to_list(matrix)
    # generate a random index
    random_index_1 = rand.randint(0, SIZE ** 2 - 1)
    random_index_2 = rand.randint(0, SIZE ** 2 - 1)
    # change the value in the chosen index
    temp = binary[random_index_1]
    binary[random_index_1] = binary[random_index_2]
    binary[random_index_2] = temp
    matrix = list_to_mat(binary)
    return fix_mat(matrix)


def main():
    global SIZE, TOTAL_RESTRICTIONS, INITIALS, INEQUALITIES, GENERATION, CURRENT_POPULATION, FIRST_MIN, BEST_MIN, BEST_MAT, MIN_LIST, AVG_LIST
    algorithm = get_algorithm()
    f = open(sys.argv[1])
    lines = f.readlines()  # read all file lines
    curr_line = 0  # first 'size' line
    SIZE = int(lines[curr_line])
    curr_line += 1  # move to line 2 - number of initial numbers

    # create INITIALS dictionary:
    for i in range(int(lines[curr_line])):
        INITIALS[(int(lines[i + 1 + curr_line][0]), int(lines[i + 1 + curr_line][2]))] = \
            int(lines[i + 1 + curr_line][4])
    curr_line += 1 + int(lines[curr_line])  # move to 'number of operators' line

    # create restrictions SYMBOLS dictionary:
    for i in range(int(lines[curr_line])):
        line = i + 1 + curr_line
        x1 = int(lines[line][0])
        y1 = int(lines[line][2])
        x2 = int(lines[line][4])
        y2 = int(lines[line][6])
        INEQUALITIES[(x1, y1)] = (x2, y2)
    # number of total restrictions on each matrix
    TOTAL_RESTRICTIONS += len(INEQUALITIES)
    print('\n')

    # initial Matrix (generation 0)
    print("STARTING POSITION OF THE GAME")
    update_matrix()
    print_mat(MATRIX)
    print(len(INEQUALITIES), "INEQUALITIES: LEFT COORDINATE > RIGHT COORDINATE")
    print(INEQUALITIES)
    print('\n')
    GENERATION += 1  # move to generation 1

    # here starts the generations loop!!!
    min_counter = 0
    probability = 0.15
    while GENERATION <= 500:
        # create current generation matrix population:
        for i in range(POPULATION_SIZE):
            if GENERATION == 1:  # we need to create the first population
                update_matrix()  # generate new random matrix for each iteration (100 matrix total)
                score = mat_score(MATRIX)  # generate score for current matrix
                CURRENT_POPULATION.append((copy.deepcopy(MATRIX), score))  # create current generation population list

        # create population_copy list based on CURRENT_POPULATION:
        population_copy = []
        # create current generation score list
        current_generation_score_list = []  # based on previous generation modifications
        if algorithm == 'Normal':
            for item in CURRENT_POPULATION:
                current_generation_score_list.append(item[1])
        if algorithm == 'Lamarck' or algorithm == 'Darwin':  # optimize the matrix
            for item in CURRENT_POPULATION:
                opt_mat = optimize(item[0], item[1])
                opt_mat_score = mat_score(opt_mat)
                population_copy.append((opt_mat, opt_mat_score))
                current_generation_score_list.append(opt_mat_score)

        min_score = min(current_generation_score_list)
        MIN_LIST.append(min_score)
        AVG_LIST.append(mean(current_generation_score_list))

        if GENERATION == 1:
            FIRST_MIN = min_score
        else:
            if min_score < BEST_MIN:
                BEST_MIN = min_score
            if min_score == BEST_MIN:
                min_counter += 1  # increase counter by 1

        index_min_score = current_generation_score_list.index(min_score)

        # find the "best" matrix in current generation
        min_item_Normal = CURRENT_POPULATION[index_min_score]
        min_mat_Normal = CURRENT_POPULATION[index_min_score][0]  # save the matrix with the lowest(=the best) score
        if algorithm == "Lamarck" or algorithm == "Darwin":
            min_item_Lamarck = population_copy[index_min_score]  # check after optimization!
            min_mat_Lamarck = population_copy[index_min_score][0]  # save the matrix with the lowest(=the best) score

        if algorithm == 'Normal':
            if min_score == 0:
                BEST_MAT = min_mat_Normal
                break
            else:
                BEST_MAT = min_mat_Normal
        if algorithm == 'Lamarck' or algorithm == 'Darwin':
            if min_score == 0:
                BEST_MAT = min_mat_Lamarck
                break
            else:
                BEST_MAT = min_mat_Lamarck

        # GENETIC ALGORITHM:
        GENERATION += 1  # move to the next generation

        new_population = []

        for i in range(10):  # copy the "best" matrix 10 times to the next generation
            if algorithm == 'Normal' or algorithm == 'Darwin':
                new_population.append(min_item_Normal)  # before optimization
            if algorithm == 'Lamarck':
                new_population.append(min_item_Lamarck)  # after optimization

        for i in range(90):
            random_index = rand.randint(0, POPULATION_SIZE-1)  # index 0:99
            if algorithm == 'Normal' or algorithm == 'Darwin':
                new_mat = crossover(CURRENT_POPULATION[random_index][0], CURRENT_POPULATION[99 - random_index][0])
                new_population.append((new_mat, mat_score(new_mat)))
            if algorithm == 'Lamarck':
                new_mat = crossover(population_copy[random_index][0], population_copy[99 - random_index][0])
                new_population.append((new_mat, mat_score(new_mat)))

        # make mutation on each matrix in chosen probability
        if min_counter == 100:
            probability = 0.30
            min_counter = 0
        for i in range(POPULATION_SIZE):  # move on all items in population
            mat = new_population[i][0]
            chance = rand.random()
            if chance <= probability:
                new_mat = mutation(mat)
                new_population[i] = (new_mat, mat_score(new_mat))

        rand.shuffle(new_population)
        CURRENT_POPULATION = new_population

    # end game
    print("for the", algorithm, "genetic algorithm:")
    print("worst fitness score possible is", TOTAL_RESTRICTIONS)
    print("best fitness score possible is 0")
    print("best fitness score in the 1st generation is", FIRST_MIN)
    print("best fitness score achieved after", GENERATION-1, "generations is", BEST_MIN, "\nwhich is", (1-BEST_MIN/TOTAL_RESTRICTIONS)*100, "% success rate")
    print_mat(BEST_MAT)

    # draw best fitness score in each generation plot
    plt.plot(MIN_LIST, label="best fitness score")
    plt.plot(AVG_LIST, label="average fitness score")
    plt.title(sys.argv[1])
    plt.xlabel('GENERATIONS')
    plt.ylabel('FITNESS SCORE (THE LOWER THE BETTER)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




