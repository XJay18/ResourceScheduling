from object.job import Job as Job1
from object.scheduler import Scheduler as Scheduler1
import os
import copy
import pickle
import argparse

INPUT_DIRECTORY = "input"
OUTPUT_DIRECTORY = "output"


def arg_parser():
    parser = argparse.ArgumentParser(description="Resource scheduling problem in Hadoop.\n"
                                                 "All the output figures and txt files are"
                                                 " saved under 'output/' directory.")
    parser.add_argument("--method", "-m", type=str, default="NMC", choices=["NMC", "OMC", "BS", "SC"],
                        help="Specifiy a method to run for resource scheduling.\n"
                             "'NMC': Naive Monte Carlo\n'OMC': Optimized Monte Carlo\n"
                             "'BS': Balanced Schedule\n'SC': Single Core Schedule")
    parser.add_argument("--input_name", "-i", type=str, default="task1_case1",
                        help="Specify the name of the input file to start the scheduling.")
    return parser.parse_args()


def run_new_task1_sample(file, run_id, version=3):
    assert version in [1, 2, 3, 4], "version should be in [1, 2, 3, 4]."
    job_list = list()
    with open(os.path.join(INPUT_DIRECTORY, file + ".txt"), "r") as f:
        contents = f.readlines()
    _, _, alpha = contents[0].strip().split(" ")
    alpha = float(alpha)
    ml = contents[1].strip()
    ml = [int(ml)]
    job_speed = contents[3].strip().split()
    job_speed = [int(i) for i in job_speed]
    for i in range(len(job_speed)):
        bsize = contents[i + 4].split(" ")
        bsize = [int(_) for _ in bsize]
        job = Job1(i, job_speed[i], bsize)
        job_list.append(job)
    best_scheduler = None
    for i in range(1):
        print(f"Trail {i}")
        scheduler = Scheduler1(alpha, 1, ml, None, seed=0)
        scheduler.load_jobs([copy.deepcopy(i) for i in job_list])
        allocation = getattr(scheduler, "allocate_jobs_v" + str(version))
        allocation()
        scheduling = getattr(scheduler, "schedule_jobs_v" + str(version))
        scheduling()
        print(scheduler.standard_summary_from_block())
        print(scheduler.standard_summary_from_core())
        if not best_scheduler:
            best_scheduler = scheduler
        elif scheduler < best_scheduler:
            best_scheduler = scheduler
    if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, file)):
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, file))
    with open(os.path.join(OUTPUT_DIRECTORY, file, f"{run_id}.txt"), "w", encoding='utf-8') as f:
        f.writelines(best_scheduler.standard_summary_from_block())
        f.writelines(best_scheduler.standard_summary_from_core())
    output = os.path.join(OUTPUT_DIRECTORY, file, f"{run_id}.png")
    # visualize
    best_scheduler.visualize(figsize=(70, 6), savefig=output)
    # best_scheduler.summary()


def run_new_task2_sample(file, run_id, version=3):
    assert version in [1, 2, 3, 4], "version should be in [1, 2, 3, 4]."
    job_list = list()
    with open(os.path.join(INPUT_DIRECTORY, file + ".txt"), "r") as f:
        contents = f.readlines()
    _, num_hosts, alpha, tspeed = contents[0].strip().split(" ")
    num_hosts = int(num_hosts)
    alpha = float(alpha)
    tspeed = int(tspeed)
    ml = contents[1].strip().split(" ")
    ml = [int(_) for _ in ml]
    job_speed = contents[3].strip().split()
    job_speed = [int(i) for i in job_speed]
    for i in range(len(job_speed)):
        bsize = contents[i + 4].strip().split(" ")
        bsize = [int(_) for _ in bsize]
        job = Job1(i, job_speed[i], bsize)
        bloc = contents[len(job_speed) + i + 4].strip().split(" ")
        bloc = [int(_) for _ in bloc]
        job.set_block_to_host(bloc)
        job_list.append(job)
    best_scheduler = None
    for i in range(1):
        print(f"Trail {i}")
        scheduler = Scheduler1(alpha, num_hosts, ml, tspeed, seed=0)
        scheduler.load_jobs([copy.deepcopy(i) for i in job_list])
        allocation = getattr(scheduler, "allocate_jobs_v" + str(version))
        if version == 3:
            allocation(std=3)
        else:
            allocation()
        scheduling = getattr(scheduler, "schedule_jobs_v" + str(version))
        scheduling()
        print(scheduler.standard_summary_from_block())
        print(scheduler.standard_summary_from_core())
        if not best_scheduler:
            best_scheduler = scheduler
        elif scheduler < best_scheduler:
            best_scheduler = scheduler
    if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, file)):
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, file))
    with open(os.path.join(OUTPUT_DIRECTORY, file, f"{run_id}.txt"), "w", encoding='utf-8') as f:
        f.writelines(best_scheduler.standard_summary_from_block())
        f.writelines(best_scheduler.standard_summary_from_core())
    output = os.path.join(OUTPUT_DIRECTORY, file, f"{run_id}.png")
    # visualize
    best_scheduler.visualize(figsize=(70, 10), savefig=output)
    # best_scheduler.summary()


def visualize(path, figsize=None):
    # visualize the best solution
    with open(os.path.join(OUTPUT_DIRECTORY, path), "rb") as f:
        sols = pickle.load(f)
    sols.visualize(figsize=figsize, savefig=os.path.join(
        OUTPUT_DIRECTORY, path.replace(".pickle", "") + ".png"))


def read_from(file, run_id):
    # print the standard summary for given pickle
    with open(os.path.join(OUTPUT_DIRECTORY, file, f"sols_{run_id}.pickle"), "rb") as f:
        sols = pickle.load(f)
    with open(os.path.join(OUTPUT_DIRECTORY, file, f"{run_id}.txt"), "w", encoding='utf-8') as f:
        f.writelines(sols.standard_summary_from_block())
        f.writelines(sols.standard_summary_from_core())
    sols.standard_summary_from_block()
    sols.standard_summary_from_core()


if __name__ == '__main__':
    arg = arg_parser()
    if arg.method == "NMC":
        if "task1" in arg.input_name:
            run_new_task1_sample(arg.input_name, "naiveMC", 1)
        elif "task2" in arg.input_name:
            run_new_task2_sample(arg.input_name, "naiveMC", 1)
        else:
            raise ValueError("'input_name' illegal. Currently only 'task[1|2]_.*?' supported.")
    elif arg.method == "OMC":
        if "task1" in arg.input_name:
            run_new_task1_sample(arg.input_name, "optimizedMC", 2)
        elif "task2" in arg.input_name:
            run_new_task2_sample(arg.input_name, "optimizedMC", 2)
        else:
            raise ValueError("'input_name' illegal. Currently only 'task[1|2]_.*?' supported.")
    elif arg.method == "BS":
        if "task1" in arg.input_name:
            run_new_task1_sample(arg.input_name, "balanced_schedule", 3)
        elif "task2" in arg.input_name:
            run_new_task2_sample(arg.input_name, "balanced_schedule", 3)
        else:
            raise ValueError("'input_name' illegal. Currently only 'task[1|2]_.*?' supported.")
    elif arg.method == "SC":
        if "task1" in arg.input_name:
            run_new_task1_sample(arg.input_name, "single_core", 4)
        elif "task2" in arg.input_name:
            run_new_task2_sample(arg.input_name, "single_core", 4)
        else:
            raise ValueError("'input_name' illegal. Currently only 'task[1|2]_.*?' supported.")
