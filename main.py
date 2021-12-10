from objects.job import Job
from objects.scheduler import Scheduler
import os
import time
import copy
import pickle

INPUT_DIRECTORY = "input"
OUTPUT_DIRECTORY = "output"


def run_new_task1_sample(file, run_id):
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
        job = Job(i, job_speed[i], bsize)
        job_list.append(job)
    best_scheduler = None
    start_time = time.perf_counter()
    for i in range(1):
        print(f"Trail {i}")
        scheduler = Scheduler(alpha, 1, ml, None, seed=0)
        scheduler.load_jobs([copy.deepcopy(i) for i in job_list])
        scheduler.allocate_jobs_v3()
        scheduler.schedule_jobs_v3()
        scheduler.standard_summary_from_block()
        scheduler.standard_summary_from_core()
        if not best_scheduler:
            best_scheduler = scheduler
        elif scheduler < best_scheduler:
            best_scheduler = scheduler
    end_time = time.perf_counter()
    if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, file)):
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, file))
    output = os.path.join(OUTPUT_DIRECTORY, file, f"sols_{run_id}.pickle")
    with open(output, "wb") as f:
        pickle.dump(best_scheduler, f)
    print(f"{'-' * 20}\nProcess time: {end_time - start_time:.4f}s\nrun id: {run_id}")
    # visualize
    visualize(output, figsize=(70, 6))
    best_scheduler.summary()


def run_new_task2_sample(file, run_id):
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
        job = Job(i, job_speed[i], bsize)
        bloc = contents[len(job_speed) + i + 4].strip().split(" ")
        bloc = [int(_) for _ in bloc]
        job.set_block_to_host(bloc)
        job_list.append(job)
    best_scheduler = None
    start_time = time.perf_counter()
    for i in range(1):
        print(f"Trail {i}")
        scheduler = Scheduler(alpha, num_hosts, ml, tspeed, seed=0)
        scheduler.load_jobs([copy.deepcopy(i) for i in job_list])
        scheduler.allocate_jobs_v3(3)
        scheduler.schedule_jobs_v3()
        scheduler.standard_summary_from_block()
        scheduler.standard_summary_from_core()
        if not best_scheduler:
            best_scheduler = scheduler
        elif scheduler < best_scheduler:
            best_scheduler = scheduler
    end_time = time.perf_counter()
    if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, file)):
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, file))
    output = os.path.join(OUTPUT_DIRECTORY, file, f"sols_{run_id}.pickle")
    with open(output, "wb") as f:
        pickle.dump(best_scheduler, f)
    print(f"{'-' * 20}\nProcess time: {end_time - start_time:.4f}s\nrun id: {run_id}")
    # visualize
    visualize(output, figsize=(70, 10))
    best_scheduler.summary()


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
    sols.standard_summary_from_block()
    sols.standard_summary_from_core()


if __name__ == '__main__':
    # run_new_task1_sample("task1_case1", "v1")
    # run_new_task2_sample("task2_case1", "v3")
    # visualize("sols_None.pickle", figsize=(70, 10), v1=False)
    read_from("task2_case1", "v4")
    pass
