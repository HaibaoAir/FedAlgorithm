import os
from subprocess import Popen
import time
import yaml


# 启动一个实验
def launch(mode, alg, dataset, net, dirichlet, init_num_class, num_epoch, env):
    # 命令
    cmd = f"python FedStream.py --alg_name {alg} --dataset_name {dataset} --net_name {net} --dirichlet {dirichlet} --init_num_class {init_num_class} --num_epoch {num_epoch} --mode {mode}"

    # 路径
    suffix = suffix = "_basic" if not mode else ""
    path_0 = "../logs/fedstream/client{}_round{}_initdata{}/{}_{}_{}/launcher{}/d{}_c{}_e{}.log".format(
        client,
        round,
        initdata,
        dataset,
        net,
        alg,
        suffix,
        dirichlet,
        init_num_class,
        num_epoch,
    )
    os.makedirs(os.path.dirname(path_0), exist_ok=True)
    with open(path_0, "w") as log_file:
        print(f"[LAUNCH] {cmd}")
        return Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )


# 套取基本信息以访问路径
with open("../config/args.yaml") as f:
    config = yaml.safe_load(f)
client = config["num_client"]
round = config["num_round"]
initdata = config["init_data_lower"]

# 参数全家桶
task_list = [
    # [
    #     [1],
    #     ["mnist", "lr", 10],
    #     ["fedavg", "fedprox", "feddyn"],
    #     [0.5, 0.8, 1],
    #     [8, 10],
    #     [20],
    # ],
    # [[1], ["fmnist", "lr", 10], ["fedavg", "fedprox", "feddyn"], [1], [10], [20]],
    # [[1], ["svhn", "cnn", 10], ["fedavg", "fedprox", "feddyn"], [1], [10], [20]],
    [
        [0, 1],
        ["cifar10", "cnn", 10],
        ["fedavg", "fedprox", "feddyn"],
        [0.1],
        [10],
        [5],
    ],
    # [
    #     [1],
    #     ["cifar100", "resnet", 100],  #
    #     ["fedavg", "fedprox", "feddyn"],
    #     [1],
    #     [100],
    #     [20],
    # ],
]

# 其他参数
gpu_list = [1, 2]
count = 0

MAX_PARALLEL = 12
process_pool = []


# 遍历所有参数组合
for task in task_list:
    for mode in task[0]:
        dataset_name = task[1][0]
        net_name = task[1][1]
        num_class = task[1][2]
        for alg_name in task[2]:
            for dirichlet in task[3]:
                for init_num_class in task[4]:
                    for num_epoch in task[5]:
                        # 设置GPU
                        gpu_id = gpu_list[count % len(gpu_list)]
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        count += 1

                        # 启动实验
                        p = launch(
                            mode,
                            alg_name,
                            dataset_name,
                            net_name,
                            dirichlet,
                            init_num_class,
                            num_epoch,
                            env,
                        )
                        process_pool.append(p)

                        # 限制最大并发
                        while len(process_pool) >= MAX_PARALLEL:
                            time.sleep(2)
                            process_pool = [
                                p for p in process_pool if p.poll() is None
                            ]  # 只保留未结束的

# 等所有实验完成
for p in process_pool:
    p.wait()

print("All experiments done.")
