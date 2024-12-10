import copy

#  remove the dynamics of grasping and use other fingers for locomotion

import mujoco
from mujoco import viewer
import sys

from controller_full import Robot
import numpy as np
import tqdm
import time

import pygad
from hand_generator_from_genes import crawling_hand
from crawling_robot_sim import crawling_robot_sim, GA_locomotion

# obj_nums = 1
pinch_one_only = 0

for i in [1000, 2000, 3000, 4000, 5000]:
    obj_nums = int(i / 1000)
    if obj_nums >= 4:
        pinch_one_only = 1
        obj_nums = 1
    print(i, '#######################################################################################')
    last_distance = -100
    locomotion_para = []
    q_grasp = []

    #  save the best one
    time_str = str(int(time.time()))
    name = str(i)
    path = '../data_records/full_test/'

    objects = False
    first_static = False  # keep the first finger static for holding an obj
    q1 = -np.array([0, 1.5, 1.5, 1.1])


    def save_data(current_distance, locomotion_para_, q_grasp_):
        global last_distance, locomotion_para, q_grasp

        if current_distance >= last_distance:
            last_distance = current_distance
            locomotion_para = locomotion_para_
            q_grasp = q_grasp_
            # print(last_distance, q_grasp)
            np.savez(path + 'GA_loco_' + name, locomotion_para=locomotion_para, q_grasp=q_grasp)


    def fitness_func(ga_instance, solution, solution_idx):
        # global last_distance, locomotion_para, q_grasp
        current_distance, locomotion_para_, q_grasp_ = GA_locomotion(solution, first_static=first_static,
                                                                     objects=objects, q1=q1, obj_nums=obj_nums,
                                                                     pinch_one_only=pinch_one_only)

        # print(current_distance, locomotion_para_, q_grasp_)
        # save_data(copy.deepcopy(current_distance), copy.deepcopy(locomotion_para_), copy.deepcopy(q_grasp_) )

        #     # save locomotion parameters
        #     name = 'solution_' + str(int(time.time()))
        #     np.save(path + 'locomotion_data/' + time_str, locomotion_para)
        return current_distance
        # return fitness


    def get_gene_space():
        # x: [fingers, dof, lengths, q0,f,a,phi]
        gene_range = [[0, 1]] * 8  # fingers
        # gene_range += [[2, 3, 4]] * 12  # dof
        gene_range += [{'low': 0.025, 'high': 0.1}] * 8  # lengths
        # gene_range += [{'low': -1, 'high': 1}] * 12 * 4  # q0
        # gene_range += [{'low': 0.1, 'high': 30}]  # frequency
        # gene_range += [{'low': -np.pi / 2, 'high': np.pi / 2}] * 12 * 4  # a
        # gene_range += [{'low': -np.pi / 2, 'high': np.pi / 2}] * 12 * 4  # alpha
        return gene_range


    num_genes = 8 * 2  # we have 7 positions to place fingers, assuming that one finger has been placed at x-axis

    num_generations = 30
    num_parents_mating = 12 * 2
    sol_per_pop = 8 * 2 * 2

    # num_generations = 5
    # num_parents_mating = 3
    # sol_per_pop = 4

    mutation_type = "random"
    mutation_percent_genes = 10

    gene_space = get_gene_space()

    last_fitness = 0

    saved_fitness = []


    def on_generation(ga_instance: pygad.GA):
        global last_fitness, saved_fitness
        print(f"Generation = {ga_instance.generations_completed}")
        print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
        print(
            f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
        last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        saved_fitness.append(ga_instance.last_generation_fitness)
        # print(ga_instance.generations_completed, ga_instance.last_generation_fitness)


    def on_fitness(self, ga_instanse, last_gen_fitness):
        print("on_fitness")


    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           fitness_func=fitness_func,
                           on_generation=on_generation,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           parallel_processing=('process', 80),
                           # save_solutions=True,
                           gene_space=gene_space,
                           gene_type=[float, 3]
                           )

    ga_instance.run()
    # ga_instance.plot_new_solution_rate()
    # print(locomotion_para, q_grasp, saved_fitness, last_distance)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # if first_static:
    #     solution[0] = 1
    #     solution[12] = 4
    print("Parameters of the best solution : {solution}".format(solution=solution))  # the best structure
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)

    current_distance, locomotion_para_, q_grasp_ = GA_locomotion(solution, first_static=first_static, objects=objects,
                                                                 q1=q1, obj_nums=obj_nums,
                                                                 pinch_one_only=pinch_one_only)

    print('Final results', q_grasp_)
    np.savez(path + 'GA_' + name, saved_fitness=current_distance, solution=solution, solution_fitness=solution_fitness,
             locomotion_para=locomotion_para_, q_grasp=q_grasp_, pinch_one_only=pinch_one_only, obj_nums=obj_nums)

    # np.save(path + 'fit_' + name, np.vstack(saved_fitness))
    # np.save(path + 'solution_' + name, np.vstack(solution))
    # np.save(path + 'locomotion_' + name, locomotion_para)
    # np.save(path + 'q_grasp_' + name, q_grasp)

    print('structure data saved to' + path + 'GA_' + name)
