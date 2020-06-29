import os
import numpy as np
import torch
from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator
from utils.Augment import augment
from utils.RolloutStore import RolloutBuffer

from torch.multiprocessing import SimpleQueue, Process, Value, Event, Barrier


def worker(worker_id,
           storage,
           ready_to_work,
           queue,
           exit_flag):
    '''
    Worker function to collect experience based on policy and store the experience in storage
    :param worker_id: id used for store the experience in storage
    :param policy: function/actor-critic
    :param storage:
    :param ready_to_work: condition to synchronize work and training
    :param queue: message queue to send episode reward to learner
    :param exit_flag: flag set by leaner to exit the job
    :param task_config_file: the task configuration file
    :return:
    '''

    print(f"Worker with Id:{worker_id} pid ({os.getpid()}) starts ...")

    steps_per_epoch = storage.block_size
    device = storage.device
    
    # Wait for start job
    print('waiting to start')
    ready_to_work.wait()

    for env in env_generator('train_small', device=device.type):
        if exit_flag.value != 1:
            x = env.reset()
            for i in range(steps_per_epoch):
                a_t = random_policy(env.action_space.n)
                # interact with environment
                o, r, d, info = env.step(a_t)
                
                # save experience
                x_aug_1 = augment(x['rgb'], storage.obs_dim[1]) # size of augmented img == size of obs in storage == size of origin img from simulator
                x_aug_2 = augment(x['rgb'], storage.obs_dim[1])
                storage.store(worker_id, x_aug_1, x_aug_2)
                
                # prepare inputs for next step
                x = o

                if d:
                    env.reset()
            storage.finish_path(worker_id)

            # print(f"Worker:{worker_id} {device} pid:{os.getpid()} begins to notify Learner Episode done")
            queue.put((worker_id))
            print(f"Worker:{worker_id} waits for next env")

            # Wait for next job
            ready_to_work.clear()
            ready_to_work.wait()

    env.close()
    print(f"Worker with pid ({os.getpid()})  finished job")

def random_policy(action_n):
    return np.random.randint(action_n)


def FakeLearner(queue, ready_to_works, exit_flag):
    _ = [e.set() for e in ready_to_works]
    exit_flag.value = 0
    id = queue.get()


if __name__ == '__main__':
    #arguement
    num_workers = 1
    steps=2048
    # obs_dim = (3, 150, 150)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.multiprocessing.set_start_method('spawn')

    # get observation dimension
    env = RoboThorEnv(config_file="config_files/NavTaskTrain.json", device='cpu')
    env.init_pos = {'x':0, 'y':0, 'z':0}
    env.init_ori = {'x':0, 'y':0, 'z':0}
    env.task.target_id = 'Apple|+01.98|+00.77|-01.75'
    env.reset()
    obs_dim = env.observation_space['rgb'].shape
    env.close()

    # Experience buffer
    storage = RolloutBuffer(obs_dim, steps, num_workers)
    storage.share_memory()

    # start multiple procRLEnvesses
    ready_to_works = [Event() for _ in range(num_workers)]
    exit_flag = Value('i', 0)
    queue = SimpleQueue()

    processes = []
    # start workers
    for worker_id in range(num_workers):
        p = Process(target=worker, args=(worker_id, storage, ready_to_works[worker_id], queue, exit_flag))
        p.start()
        processes.append(p)

    p = Process(target=FakeLearner, args=(queue, ready_to_works, exit_flag))
    p.start()
    processes.append(p)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()



