import numpy as np


class Job(object):
    def __init__(self, index, speed, bsize, bloc=None):
        self.index = index
        self.speed = speed
        self.bsize = np.array(bsize)  # block size
        self.ni = self.bsize.shape[0]  # number of data blocks
        self.bloc = bloc if bloc else [0] * self.ni  # block location
        self.allocation = None  # ni-dimension
        self.cores_to_use = None
        self.start_time = 0.
        self.duration = None  # m-dimension
        self.block_rank = np.zeros(self.ni, dtype=int)  # ni-dimension
        self.max_duration = None

    def set_block_to_host(self, bloc):
        self.bloc = bloc
        assert len(self.bloc) == len(self.bsize)

    def clear(self):
        self.allocation = None
        self.cores_to_use = None
        self.start_time = 0.
        self.duration = None
        self.block_rank = np.zeros(self.ni, dtype=int)  # ni-dimension
        self.max_duration = None

    def __str__(self):
        return f"Job {self.index} with {self.ni} blocks, speed: {self.speed}."

    def details(self) -> str:
        return f"Job {self.index}\t | #Blocks: {self.ni}\t | Speed: {self.speed}\t | BSize: {self.bsize}\n" \
               f"allocation: {self.allocation}\t | start: {self.start_time}\t | duration: {self.duration}\n"


if __name__ == '__main__':
    pass
