import os
import numpy as np
from objects.job import Job
from collections import namedtuple
import matplotlib.pyplot as plt

Task = namedtuple("Task", "job block t_start t_end r_end")
Core = namedtuple("Core", "host core")


class Scheduler(object):
    """
    Resource scheduler for both single- and multi-hosts problem.
    """

    def __init__(self, alpha, q, ml, tspeed, seed=None):
        """
        Constructor for resource scheduler.

        Args:
            alpha: The decay coefficient for parallel processing.
            q: The number of hosts available.
            ml: A list specified the number of cores available on each host.
            tspeed: The transmission speed between different hosts.
            seed: The seed used for randomization.
        """
        self.alpha = alpha
        self.q = q  # number of hosts
        self.ml = ml  # numbers of cores in different hosts, q-dimension array
        self.m = sum(self.ml)  # number of cores
        self.tspeed = tspeed  # transmission speed

        self.timelines = np.zeros(shape=self.m)
        self.jobs = list()
        self.tasks = dict({i: list() for i in range(self.m)})  # record the task list
        self.ncore_job = dict({i + 1: list() for i in range(self.m)})
        self.core_to_host = dict()
        self.__init_core_to_host()

        self.seed = seed
        np.random.seed(seed)

    def __gt__(self, other):
        return np.max(self.timelines) > np.max(other.timelines)

    def __eq__(self, other):
        return np.max(self.timelines) == np.max(other.timelines)

    def __init_core_to_host(self):
        cnt = 0
        for i in range(self.q):
            for j in range(self.ml[i]):
                self.core_to_host.update({cnt: Core(host=i, core=j)})
                cnt += 1

    def standard_summary_from_block(self):
        print(f"\nTask{2 if self.tspeed else 1} Solution (Block Perspective) of Team No.30:\n")
        total_response_time = 0.
        for i in self.jobs:
            finish_time = i.start_time + i.max_duration
            total_response_time += finish_time
            print(f"Job{i.index} obtains {len(i.cores_to_use)} cores (speed={i.speed:.2f}) "
                  f"and finishes at time {finish_time:.6f}: ")
            for j in range(i.ni):
                core = self.core_to_host.get(i.allocation[j])
                if self.tspeed and core.host != i.bloc[j]:
                    trans_time = i.bsize[j] / self.tspeed
                    trans_time = round(trans_time, 2)
                else:
                    trans_time = None
                print(f"\tBlock{j}: H{core.host}, C{core.core}, R{i.block_rank[j]} "
                      f"(trans time={trans_time}, proc time={i.bsize[j] / i.speed:.2f}) ")
            print()
        print(f"The maximum finish time: {np.max(self.timelines):.6f}")
        print(f"The total response time: {total_response_time:.6f}")

    def standard_summary_from_core(self):
        running_time = 0.
        max_timelines = np.max(self.timelines)
        print(f"\nTask{2 if self.tspeed else 1} Solution (Core Perspective) of Team No.30:\n")
        flag = [False for _ in range(self.q)]
        for j in range(self.m):
            core = self.core_to_host.get(j)
            if not flag[core.host]:
                start = sum(self.ml[:core.host])
                all_cores = np.array(range(start, start + self.ml[core.host]))
                cores_finish_time = list()
                for c in all_cores:
                    cores_finish_time.append(self.tasks.get(c)[-1].r_end)
                print(f"Host{core.host} finishes at time {np.max(cores_finish_time):.6f}:\n")
                flag[core.host] = True
            print(f"\tCore{core.core} has {len(self.tasks.get(j)) + 1} tasks and "
                  f"finishes at time {self.tasks.get(j)[-1].r_end:.6f}:")
            for k in self.tasks.get(j):
                running_time += k.r_end - k.t_start
                trans_description = f"{k.t_start:.2f} to {k.t_end:.2f}" if k.t_end != k.t_start else f"None"
                print(f"\t\tJ{k.job:02}, B{k.block:02}, trans time {trans_description}, "
                      f"proc time {k.t_end:.2f} to {k.r_end:.2f}")
            print()
        print(f"\nThe maximum finish time of hosts: {max_timelines:.6f}")
        print(f"The total efficacious running time: {running_time:.6f}")
        print(f"Utilization rate: {running_time / self.m / max_timelines:.6f}")

    def load_jobs(self, jobs):
        for _ in jobs:
            assert isinstance(_, Job), f"{str(_)} is not a Job."
            self.jobs.append(_)

    def allocate_jobs_v4(self):
        """
        Use single core for any job.
        """
        for job in self.jobs:
            job.allocation = np.zeros(job.ni, dtype=int)
            job.cores_to_use = np.unique(job.allocation)
            # record the number of cores to use
            self.ncore_job.get(len(job.cores_to_use)).append(job.index)
            time_vec = np.zeros(shape=self.m)
            ei = job.cores_to_use.shape[0]
            g_ei = 1. - self.alpha * (ei - 1)
            job.speed = g_ei * job.speed
            for i in job.cores_to_use:
                # data block to run in the i-th core
                blocks_to_run = np.where(job.allocation == i)[0]
                total_size = sum(job.bsize[blocks_to_run])
                run_time = total_size / job.speed
                time_vec[i] = run_time
            job.duration = time_vec
            job.max_duration = np.max(time_vec)

    def allocate_jobs_v3(self, std=4):
        """
        First compute the standard deviation of block sizes for a job under parallel processing,
        then if the minimum standard deviation is lower than 'std', we consider using parallel
        processing for the job, otherwise, the job will run on single core.
        """
        for job in self.jobs:
            allocate_list = list()
            std_list = list()
            sorted_block_id = np.argsort(job.bsize)[::-1]
            # find the most balanced allocation
            for i in range(2, self.m + 1):
                temp_cores = np.zeros(i, dtype=int)
                allocate_vec = np.zeros(job.ni, dtype=int)
                for j in sorted_block_id:
                    k = np.argmin(temp_cores)
                    temp_cores[k] += job.bsize[j]
                    allocate_vec[j] = k
                allocate_list.append(allocate_vec)
                std_list.append(np.std(temp_cores))
            min_std_id = np.argmin(std_list).item()
            # ei = min_std_id + 2  # num of cores used
            # g_ei = 1. - self.alpha * (ei - 1)
            # job.allocation = allocate_list[-1]
            # if std_list[min_std_id] / (g_ei * job.speed) < 0.2:
            if std_list[min_std_id] < std:
                job.allocation = allocate_list[min_std_id]
            else:
                job.allocation = np.zeros(job.ni, dtype=int)
            job.cores_to_use = np.unique(job.allocation)
            # record the number of cores to use
            self.ncore_job.get(len(job.cores_to_use)).append(job.index)
            time_vec = np.zeros(shape=self.m)
            ei = job.cores_to_use.shape[0]
            g_ei = 1. - self.alpha * (ei - 1)
            job.speed = g_ei * job.speed
            for i in job.cores_to_use:
                # data block to run in the i-th core
                blocks_to_run = np.where(job.allocation == i)[0]
                total_size = sum(job.bsize[blocks_to_run])
                run_time = total_size / job.speed
                time_vec[i] = run_time
            job.duration = time_vec
            job.max_duration = np.max(time_vec)

    def allocate_jobs_v2(self):
        """
        Randomly set the number of cores used for a job.
        """
        for job in self.jobs:
            # random choosing number of cores
            num_cores = np.random.randint(1, min(job.ni, self.m) + 1)
            # record the number of cores to use
            self.ncore_job.get(num_cores).append(job.index)
            temp_cores = np.zeros(shape=num_cores)
            allocate_vec = np.zeros(shape=job.ni, dtype=int)
            sorted_block_id = np.argsort(job.bsize)[::-1]  # descending
            # satisfied the optimal situation under given number of cores
            flip = np.random.random() > 0.5
            for i in sorted_block_id:
                j = np.argmin(temp_cores)
                temp_cores[j] += job.bsize[i]
                if 2 * num_cores > self.m:
                    allocate_vec[i] = j
                elif 2 * num_cores < self.m:
                    allocate_vec[i] = self.m - j - 1
                else:
                    if flip:
                        allocate_vec[i] = self.m - j - 1
                    else:
                        allocate_vec[i] = j
            job.allocation = allocate_vec
            cores_to_use = np.unique(allocate_vec)
            job.cores_to_use = cores_to_use
            time_vec = np.zeros(shape=self.m)
            ei = cores_to_use.shape[0]
            g_ei = 1. - self.alpha * (ei - 1)
            job.speed = g_ei * job.speed
            for i in cores_to_use:
                # data block to run in the i-th core
                blocks_to_run = np.where(allocate_vec == i)[0]
                total_size = sum(job.bsize[blocks_to_run])
                run_time = total_size / job.speed
                time_vec[i] = run_time
            job.duration = time_vec
            job.max_duration = np.max(time_vec)

    def allocate_jobs_v1(self):
        """
        Randomly allocate the cores for a job.
        """
        for job in self.jobs:
            allocate_vec = np.random.randint(0, self.m, size=job.ni)
            # # # # # # # #
            job.allocation = allocate_vec
            time_vec = np.zeros(shape=self.m)
            cores_to_use = np.unique(allocate_vec)
            job.cores_to_use = cores_to_use
            ei = cores_to_use.shape[0]
            g_ei = 1. - self.alpha * (ei - 1)
            job.speed = g_ei * job.speed
            for i in cores_to_use:
                # data block to run in the i-th core
                blocks_to_run = np.where(allocate_vec == i)[0]
                total_size = sum(job.bsize[blocks_to_run])
                run_time = total_size / job.speed
                if not self.tspeed:
                    time_vec[i] = run_time
                    continue
                transmission_size = 0.
                for j in blocks_to_run:
                    if job.bloc[j] != self.core_to_host.get(i).host:
                        # transmission necessary
                        transmission_size += job.bsize[j]
                transmission_time = transmission_size / self.tspeed
                time_vec[i] = run_time + transmission_time
            job.duration = time_vec
            job.max_duration = np.max(time_vec)

    def adjust_cores(self, job, n_cores):
        if self.tspeed:
            self.__adjust_cores_w_trans(job, n_cores)
        else:
            self.__adjust_cores_wo_trans(job, n_cores)

    def __adjust_cores_wo_trans(self, job, n_cores):
        """
        Adjusting the allocated cores for a job without considering the transmission time for
        a better scheduling result.

        Args:
            job: The job considered.
            n_cores: The number of cores originally assigned to the job.
        """
        ascending_timeline = np.argsort(self.timelines)
        chosen_cores = ascending_timeline[:n_cores]
        c_t_u = job.cores_to_use
        adjust_map = {c_t_u[_]: chosen_cores[_] for _ in range(n_cores)}
        changed_alloc_vec = [adjust_map.get(_) for _ in job.allocation]
        changed_duration = np.zeros(self.m)
        for _ in c_t_u:
            changed_duration[adjust_map.get(_)] = job.duration[_]
        job.allocation = np.array(changed_alloc_vec)
        job.cores_to_use = np.unique(changed_alloc_vec)
        job.duration = changed_duration
        job.start_time = self.timelines[chosen_cores[-1]]
        self.update_rank(job)

    def __adjust_cores_w_trans(self, job, n_cores):
        """
        Adjusting the allocated cores for a job with the transmission time considered for a
        better scheduling result.

        Args:
            job: The job considered.
            n_cores: The number of cores originally assigned to the job.
        """
        from itertools import combinations
        chosen_cores_list = list()
        start_time_list = list()
        for chosen_cores in combinations(range(self.m), n_cores):
            chosen_cores_list.append(chosen_cores)
            c_t_u = job.cores_to_use
            cores_start_time = self.timelines[np.array(chosen_cores)]
            adjust_map = {c_t_u[_]: chosen_cores[_] for _ in range(n_cores)}
            changed_alloc_vec = np.array([adjust_map.get(_) for _ in job.allocation])
            for i, k in enumerate(chosen_cores):
                # data block to run in the i-th core
                blocks_to_run = np.where(changed_alloc_vec == k)[0]
                transmission_size = 0.
                for l in blocks_to_run:
                    if job.bloc[l] != self.core_to_host.get(k).host:
                        # transmission necessary
                        transmission_size += job.bsize[l]
                transmission_time = transmission_size / self.tspeed
                cores_start_time[i] += transmission_time
            start_time_list.append(np.max(cores_start_time))
        # select the best choice
        best_choice = np.argmin(start_time_list)
        chosen_cores = chosen_cores_list[best_choice]
        # update the job information
        c_t_u = job.cores_to_use
        adjust_map = {c_t_u[_]: chosen_cores[_] for _ in range(n_cores)}
        changed_alloc_vec = [adjust_map.get(_) for _ in job.allocation]
        changed_duration = np.zeros(self.m)
        for _ in c_t_u:
            changed_duration[adjust_map.get(_)] = job.duration[_]
        job.allocation = np.array(changed_alloc_vec)
        job.cores_to_use = np.unique(changed_alloc_vec)
        job.duration = changed_duration
        for k in job.cores_to_use:
            # data block to run in the i-th core
            blocks_to_run = np.where(job.allocation == k)[0]
            transmission_size = 0.
            for l in blocks_to_run:
                if job.bloc[l] != self.core_to_host.get(k).host:
                    # transmission necessary
                    transmission_size += job.bsize[l]
            transmission_time = transmission_size / self.tspeed
            job.duration[k] += transmission_time
        job.max_duration = np.max(job.duration)
        job.start_time = np.max(self.timelines[np.array(chosen_cores)])
        self.update_rank(job)

    def schedule_jobs_v4(self):
        """
        Use 'job scheduling algorithm' to order the jobs.
        """
        processed = np.zeros(len(self.jobs), dtype=int)
        # sort the lists non-increasing
        for i in range(1, self.m + 1):
            self.ncore_job.update({i: sorted(self.ncore_job.get(i), key=lambda x: -self.jobs[x].max_duration)})
        for i in range(self.m, 0, -1):
            for j in self.ncore_job.get(i):
                if processed[j]:
                    continue
                # adjust the allocated cores
                self.adjust_cores(self.jobs[j], i)
                self.update_timeline(self.jobs[j])
                processed[j] += 1
        print(f"proc: {processed}")

    def schedule_jobs_v3(self):
        """
        Sort the jobs according to their number of cores used, each time we arrange a job, we
        consider arranging several 1-core jobs to balance the overall timeline.
        """
        processed = np.zeros(len(self.jobs), dtype=int)
        # sort the lists non-increasing
        for i in range(1, self.m + 1):
            self.ncore_job.update({i: sorted(self.ncore_job.get(i), key=lambda x: -self.jobs[x].max_duration)})
        for i in range(self.m, 0, -1):
            for j in self.ncore_job.get(i):
                if processed[j]:
                    continue
                # adjust the allocated cores
                self.adjust_cores(self.jobs[j], i)
                self.update_timeline(self.jobs[j])
                processed[j] += 1
                if i == self.m:
                    continue
                flip_i = 1
                # currently only consider 1-matching
                for flip_j in self.ncore_job.get(flip_i):
                    if processed[flip_j]:
                        continue
                    # adjust the allocated cores
                    self.adjust_cores(self.jobs[flip_j], flip_i)
                    self.update_timeline(self.jobs[flip_j])
                    processed[flip_j] += 1
                    if np.std(self.timelines) < 4:
                        break
        print(f"proc: {processed}")

    def schedule_jobs_v2(self):
        """
        Sort the jobs according to their number of cores used, each time we arrange a job, we
        consider arranging a complementary job to balance the timeline.
        """
        processed = np.zeros(len(self.jobs), dtype=int)
        # sort the lists non-increasing
        for i in range(1, self.m + 1):
            self.ncore_job.update({i: sorted(self.ncore_job.get(i), key=lambda x: -self.jobs[x].max_duration)})
        for i in range(self.m, 0, -1):
            for j in self.ncore_job.get(i):
                if processed[j]:
                    continue
                if i > self.m // 2:
                    self.jobs[j].start_time = np.max(self.timelines[self.jobs[j].cores_to_use])
                    self.update_rank(self.jobs[j])
                    self.update_timeline(self.jobs[j])
                    processed[j] += 1
                    if i == self.m:
                        continue
                    flip_i = self.m - i
                    # currently only consider 1-matching
                    for flip_j in self.ncore_job.get(flip_i):
                        if processed[flip_j]:
                            continue
                        start_time = np.max(self.timelines[self.jobs[flip_j].cores_to_use])
                        # do not insert when already overflow
                        if start_time > np.max(self.timelines):
                            break
                        self.jobs[flip_j].start_time = start_time
                        self.update_rank(self.jobs[flip_j])
                        self.update_timeline(self.jobs[flip_j])
                        processed[flip_j] += 1
                        break
                else:
                    # adjust the allocated cores
                    self.adjust_cores(self.jobs[j], i)
                    self.update_timeline(self.jobs[j])
                    processed[j] += 1
        print(f"proc: {processed}")

    def schedule_jobs_v1(self):
        """
        Sort the jobs randomly.
        """
        jobs_id = np.random.permutation(range(len(self.jobs)))
        # # # # # # # #
        last_job = self.jobs[jobs_id[0]]
        self.update_timeline(last_job)
        for i in jobs_id:
            self.jobs[i].start_time = np.max(self.timelines[self.jobs[i].cores_to_use])
            self.update_rank(self.jobs[i])
            self.update_timeline(self.jobs[i])

    def update_timeline(self, job):
        cores_to_use = job.cores_to_use
        for i in cores_to_use:
            self.timelines[i] = job.start_time + job.max_duration

    def update_rank(self, job):
        # update rank
        for i in job.cores_to_use:
            # data block to run in the i-th core
            accumulate_time = 0.
            blocks_to_run = np.where(job.allocation == i)[0]
            for j in blocks_to_run:
                job.block_rank[j] = len(self.tasks.get(i))
                block_duration = job.bsize[j] / job.speed
                if self.tspeed and self.core_to_host.get(i).host != job.bloc[j]:
                    trans_time = job.bsize[j] / self.tspeed
                else:
                    trans_time = 0.
                # append task to specific core
                self.tasks.get(i).append(
                    Task(job=job.index,
                         block=j,
                         t_start=job.start_time + accumulate_time,
                         t_end=job.start_time + accumulate_time + trans_time,
                         r_end=job.start_time + accumulate_time + trans_time + block_duration))
                accumulate_time += block_duration + trans_time

    def summary(self, detailed=False):
        max_tf = np.max(self.timelines)
        string = f"Last finished time: {max_tf}, seed: {self.seed}\n"
        if detailed:
            for _ in self.jobs:
                string += _.details() + "\n"
        print(string)

    def dump(self, output, detailed=True):
        max_tf = np.max(self.timelines)
        string = f"Last finished time: {max_tf}, seed: {self.seed}\n"
        if detailed:
            for _ in self.jobs:
                string += _.details() + "\n"
        with open(os.path.join(output, "log.txt"), "a") as f:
            f.writelines(string)

    def visualize(self, figsize=None, savefig=None):
        fig, ax = plt.subplots(figsize=figsize)
        y = ["h" + str(i) + " c" + str(j) for i in range(self.q) for j in range(self.ml[i])]
        for _ in sorted(self.jobs, key=lambda x: x.index):
            ax.barh(y, _.duration, 0.6, left=_.start_time, alpha=0.75, label=f"{_.index}")
            for i in range(self.m):
                if _.duration[i] == 0.:
                    continue
                ax.text(_.start_time + _.duration[i], y[i],
                        str(round(_.start_time + _.duration[i], 3)),
                        fontsize=10, fontweight='bold',
                        color='grey')
                blocks_to_run = np.where(_.allocation == i)[0]
                b_description = ""
                for j in blocks_to_run:
                    if _.bloc[j] != self.core_to_host.get(i).host:
                        b_description += str(j) + "* "
                    else:
                        b_description += str(j) + " "
                ax.text(_.start_time + 0.2, i - 0.2,
                        f"{_.index} b " + b_description,
                        fontsize=10, fontweight='bold',
                        color='white')

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Remove axes splines
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax.grid(b=True, color='grey', axis='x',
                linestyle='-.', linewidth=1,
                alpha=0.8)

        ax.legend(loc="best")
        plt.xlabel("Timeline", fontdict=dict(fontsize=16))
        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()
