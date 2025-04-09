import os
from subprocess import Popen
import time

# 参数组合
dirichlet_list = [0.5, 0.6, 0.7, 0.8]
num_epoch_list = [20]
init_num_class_list = [7, 8, 9, 10]

# 控制最大同时运行的进程数
MAX_PARALLEL = 32
process_pool = []


# 启动一个实验
def launch(dirichlet, num_epoch, init_num_class, env):
    cmd = f"python FedStream.py --dirichlet {dirichlet} --num_epoch {num_epoch} --init_num_class {init_num_class}"
    log_name = f"d{dirichlet}_e{num_epoch}_c{init_num_class}.log"
    log_file = open(f"../logs/launcher_logs/{log_name}", "w")
    print(f"[LAUNCH] {cmd}")
    return Popen(
        cmd,
        shell=True,
        stdout=log_file,
        stderr=log_file,
        env=env,
    )


# 遍历所有参数组合
gpu_list = [0, 1, 2]
count = 0
for dirichlet in dirichlet_list:
    for num_epoch in num_epoch_list:
        for init_num_class in init_num_class_list:
            # 设置GPU
            gpu_id = gpu_list[count % len(gpu_list)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            count += 1

            # 启动实验
            p = launch(
                dirichlet,
                num_epoch,
                init_num_class,
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
