import os
from subprocess import Popen
import time
import yaml

# 套取基本信息以访问路径
with open("../config/args.yaml") as f:
    config = yaml.safe_load(f)
client = config["num_client"]
round = config["num_round"]
initdata = config["init_data_lower"]

alg_list = ["fedavg", "fedprox", "feddyn"]
dataset_list = ["cifar10"]
net_list = ["cnn"]
dirichlet_list = [0.5, 0.8, 1]
init_num_class_list = [10]
num_epoch_list = [20]

# # 参数组合
# alg_list = ["fedavg"]
# dataset_list = ["cifar10"]
# net_list = ["cnn"]
# dirichlet_list = [0.5, 0.8, 1]
# init_num_class_list = [10]
# num_epoch_list = [20, 30]

# 控制最大同时运行的进程数
MAX_PARALLEL = 20
process_pool = []


# 启动一个实验
def launch(alg, dataset, net, dirichlet, init_num_class, num_epoch, env):
    # 命令
    cmd = f"python FedStream.py --alg_name {alg} --dataset_name {dataset} --net_name {net} --dirichlet {dirichlet} --init_num_class {init_num_class} --num_epoch {num_epoch}"

    # 路径
    path_0 = "../logs/fedstream/client{}_round{}_initdata{}/{}_{}_{}/launcher/d{}_c{}_e{}.log".format(
        client,
        round,
        initdata,
        dataset,
        net,
        alg,
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


# 遍历所有参数组合
gpu_list = [0]
count = 0
for alg in alg_list:
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        net = net_list[i]
        for dirichlet in dirichlet_list:
            for init_num_class in init_num_class_list:
                for num_epoch in num_epoch_list:
                    # 设置GPU
                    gpu_id = gpu_list[count % len(gpu_list)]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    count += 1

                    # 启动实验
                    p = launch(
                        alg,
                        dataset,
                        net,
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
