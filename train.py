import argparse
import torch
import torch.nn as nn
from torch.multiprocessing import SimpleQueue, Process, Value, Event, Barrier
import random

from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator

from Worker import worker
from Learner import learner
from Model import CAM, Encoder
from utils.RolloutStore import RolloutBuffer


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Representation Learning')

    parser.add_argument('--epochs', default=100000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--mini-batch', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--model-path', type=str, default=None)

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # get observation dimension
    env = RoboThorEnv(config_file="config_files/NavTaskTrain.json", device='cpu')
    env.init_pos = {'x':0, 'y':0, 'z':0}
    env.init_ori = {'x':0, 'y':0, 'z':0}
    env.task.target_id = 'Apple|+01.98|+00.77|-01.75'
    env.reset()
    obs_dim = env.observation_space['rgb'].shape
    env.close()

    # Experience buffer
    storage = RolloutBuffer(obs_dim, args.steps, args.num_workers)
    storage.share_memory()

    model = CAM(Encoder).to(device)
    # print('>>>>>>>>>>>>>>>>>>>', model)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    # start multiple procRLEnvesses
    ready_to_works = [Event() for _ in range(args.num_workers)]
    exit_flag = Value('i', 0)
    queue = SimpleQueue()

    processes = []
    # start workers
    for worker_id in range(args.num_workers):
        p = Process(target=worker, args=(worker_id, storage, ready_to_works[worker_id], queue, exit_flag))
        p.start()
        processes.append(p)

    # p = Process(target=learner, args=(model, storage, optimizer, criterion, args.mini_batch, args.epochs, args.num_iter, args.num_workers, queue, ready_to_works, exit_flag))
    # p.start()
    # processes.append(p)
    learner(model, storage, args.mini_batch, args.epochs, args.num_iter, args.num_workers, device, queue, ready_to_works, exit_flag)
 

    for p in processes:
        print(" >>>>>>>>>>>>>>>>>>>>> process ", p.pid, " joined")
        p.join()












