import os
import sys

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm
import argparse
import yaml

sys.path.append("../")
from Clients import Client_Group
from model.mnist import MNIST_LR, MNIST_MLP, MNIST_CNN
from model.fmnist import FMNIST_LR, FMNIST_MLP, FMNIST_CNN
from model.cifar10 import Cifar10_CNN
from model.SVHN import SVHN_CNN
from model.cifar100 import Cifar100_ResNet18
from model.TinyImageNet import TinyImageNet_ResNet18


class Server(object):
    def __init__(self, args):
        # 初始化客户端
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dataset_name = args["dataset"]
        self.num_client = args["num_client"]
        self.num_sample = args["num_sample"]
        self.net_name = args["model"]
        self.alg_name = args["alg"]
        self.learning_rate = args["learning_rate"]
        self.eta = self.learning_rate
        self.init_data_net()

        # 训练超参数
        self.num_round = args["num_round"]
        self.num_epoch = args["num_epoch"]
        self.batch_size = args["batch_size"]
        self.eval_freq = args["eval_freq"]
        self.path_prefix = args["path_prefix"]

        self.init_data_lower = args["init_data_lower"]
        self.init_data_upper = args["init_data_upper"]

        # 初始化data_matrix[K,T]
        self.data_origin_init = [
            random.randint(self.init_data_lower, self.init_data_upper)
            for _ in range(self.num_client)
        ]
        self.data_matrix_init = []
        for k in range(self.num_client):
            data_list = [self.data_origin_init[k]]
            for t in range(1, self.num_round):
                data = random.randint(50, 100)
                data_list.append(data)
            self.data_matrix_init.append(data_list)

        # 初始化phi_list[T]
        self.phi_list_init = []
        for t in range(self.num_round):
            phi = random.random() * 60
            self.phi_list_init.append(phi)

        # 背景超参数
        self.delta_list = np.array([args["delta"]] * self.num_client)  # [K]
        self.psi = args["psi"]
        self.alpha_list = [args["alpha"]] * self.num_client  # 收集数据的价格
        self.beta_list = [
            args["beta"]
        ] * self.num_client  # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.sigma = args["sigma"]

        self.kappa_1 = args["kappa_1"]
        self.kappa_2 = args["kappa_2"]
        self.kappa_3 = args["kappa_3"]
        self.kappa_4 = args["kappa_4"]
        self.gamma = args["gamma"]
        self.new_gamma = args["new_gamma"]

        self.reward_lb = args["reward_lb"]  # 反正不看过程只看结果，原来1-100得50
        self.reward_ub = args["reward_ub"]
        self.theta_lb = args["theta_lb"]
        self.theta_ub = args["theta_ub"]
        self.n_calls = args["n_calls"]
        self.base_estimator = args["base_estimator"]
        self.noise = args["noise"]
        self.acq_func = args["acq_func"]
        self.xi = args["xi"]
        self.random_state = args["random_state"]

        self.fix_eps_1 = args["fix_eps_1"]
        self.fix_eps_2 = args["fix_eps_2"]
        self.fix_eps_3 = args["fix_eps_3"]
        self.fix_max_iter = args["fix_max_iter"]

    def init_data_net(self):
        self.client_group = Client_Group(
            self.dev,
            self.num_client,
            self.dataset_name,
            args["num_class"],
            args["init_num_class"],
            args["dirichlet"],
            self.net_name,
            self.learning_rate,
        )
        self.test_data_list = self.client_group.test_data_list

        # 定义net
        self.net = None
        if self.dataset_name == "mnist":
            if self.net_name == "lr":
                self.net = MNIST_LR()
            elif self.net_name == "mlp":
                self.net = MNIST_MLP()
            elif self.net_name == "cnn":
                self.net = MNIST_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "fmnist":
            if self.net_name == "lr":
                self.net = FMNIST_LR()
            elif self.net_name == "mlp":
                self.net = FMNIST_MLP()
            elif self.net_name == "cnn":
                self.net = FMNIST_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "cifar10":
            if self.net_name == "cnn":
                self.net = Cifar10_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "svhn":
            if self.net_name == "cnn":
                self.net = SVHN_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "cifar100":
            if self.net_name == "cnn":
                self.net = Cifar100_ResNet18()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "tinyimagenet":
            if self.net_name == "cnn":
                self.net = TinyImageNet_ResNet18()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        else:
            raise NotImplementedError("{}".format(self.net_name))

        self.net.to(self.dev)

    def estimate_D(self, phi_list, reward, theta):
        """
        函数1：给定reward和theta，估计D
        """
        # 初始化数据矩阵
        data_matrix = self.data_matrix_init

        for idc in range(self.fix_max_iter):
            # 计算增量矩阵[K,T]
            increment_matrix = []
            for k in range(self.num_client):
                increment_list = []
                for t in range(0, self.num_round - 1):
                    item = 0
                    for tau in range(t + 1, self.num_round):
                        item1 = pow(theta, tau - t - 1)
                        item2 = reward / (self.delta_list[k] * phi_list[tau])
                        item3 = 2 * self.beta_list[k] * data_matrix[k][tau]
                        item += item1 * (item2 - item3)
                    increment = 1 / (2 * self.alpha_list[k]) * item
                    if increment <= 0:
                        print("dual")
                    increment = max(0, increment)  # 好哇好
                    increment_list.append(increment)
                increment_list.append(0)
                increment_matrix.append(increment_list)

            # 新的数据矩阵[K, T]
            next_data_matrix = []
            for k in range(self.num_client):
                next_data_list = [np.array(self.data_origin_init[k])]
                for t in range(1, self.num_round):
                    next_data = (
                        theta * next_data_list[t - 1] + increment_matrix[k][t - 1]
                    )
                    next_data_list.append(next_data)
                next_data_matrix.append(next_data_list)

            # 判断收敛
            flag = 0
            for k in range(self.num_client):
                for t in range(1, self.num_round):
                    if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.fix_eps_1:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 1:
                data_matrix = next_data_matrix
            else:
                # print('triumph1, count = {}'.format(idc))
                stale_matrix = [[1] * self.num_round] * self.num_client
                for k in range(self.num_client):
                    for t in range(1, self.num_round):
                        stale_matrix[k][t] = (
                            stale_matrix[k][t - 1]
                            * theta
                            * next_data_matrix[k][t - 1]
                            / next_data_matrix[k][t]
                            + 1
                        )
                return (
                    np.array(increment_matrix),
                    np.array(next_data_matrix),
                    np.array(stale_matrix),
                )

        print("failure1")
        return np.array(next_data_matrix)

    def estimate_reward_theta(self, phi_list):
        """
        函数2：给定phi_list，估计reward和theta
        """

        def cost_fn(var):
            reward, theta = var

            # 计算单列和
            increment_matrix, data_matrix, stale_matrix = self.estimate_D(
                phi_list, reward, theta
            )
            data_list = []  # [T]
            for t in range(self.num_round):
                data = 0
                for k in range(self.num_client):
                    data += data_matrix[k][t]
                data_list.append(data)

            cost = 0
            for t in range(self.num_round):
                delta_sum = 0
                stale_sum = 0
                for k in range(self.num_client):
                    delta_sum += data_matrix[k][t] * (self.delta_list[k] ** 2)
                    stale_sum += (
                        data_matrix[k][t] * stale_matrix[k][t] * (self.sigma**2)
                    )

                item_1 = (
                    pow(self.kappa_1, self.num_round - 1 - t)
                    * self.kappa_2
                    * self.num_client
                    * (self.psi**2)
                    / data_list[t]
                )
                item_2 = (
                    pow(self.kappa_1, self.num_round - 1 - t)
                    * self.kappa_3
                    * stale_sum
                    / data_list[t]
                )
                item_3 = (
                    pow(self.kappa_1, self.num_round - 1 - t)
                    * self.kappa_4
                    * delta_sum
                    / data_list[t]
                )
                item = (1 - self.gamma) * (
                    item_1 + item_2 + item_3
                ) + self.gamma * reward
                cost += item
            return cost

        result = gp_minimize(
            func=cost_fn,  # 目标函数
            dimensions=[
                Real(self.reward_lb, self.reward_ub),
                Real(self.theta_lb, self.theta_ub),
            ],  # 搜索空间
            n_calls=self.n_calls,  # 迭代次数，默认100
            base_estimator=self.base_estimator,  # 代理模型
            noise=self.noise,
            acq_func=self.acq_func,  # 采集函数
            xi=self.xi,  # 采集函数探索程度，默认0.01
            random_state=self.random_state,  # 随机种子
        )
        return result.x, result.fun, result.x_iters

    def estimate_phi(self):
        """
        函数3：给定D，估计phi
        """

        # 初始化phi_list
        phi_list = self.phi_list_init
        phi_hist = []
        reward_hist = []
        theta_hist = []
        max_diff_hist = []

        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            var, res, hist = self.estimate_reward_theta(phi_list)
            reward, theta = var
            increment_matrix, data_matrix, _ = self.estimate_D(phi_list, reward, theta)
            print("******************************************************")
            print("{}, {}, {}, {}, {}".format(idc, hist[-10:], reward, theta, res))
            # print("{}".format(data_matrix))

            next_phi_list = np.sum(
                data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)),
                axis=0,
            )
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            print("max_diff_phi:{}".format(max_diff))

            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list
                phi_hist.append(phi_list[-1])
                reward_hist.append(reward)
                theta_hist.append(theta)
                max_diff_hist.append(max_diff)

                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.set_xlabel("iterations")
                ax1.set_ylabel("phi")
                ax1.plot(phi_hist, "k-")

                ax2 = fig.add_subplot(2, 1, 2)
                ax2.set_xlabel("iterations")
                ax2.set_ylabel("reward")
                ax2.spines["left"].set_edgecolor("C0")
                ax2.yaxis.label.set_color("C0")
                ax2.tick_params(axis="y", colors="C0")
                line_2 = ax2.plot(
                    reward_hist, color="C0", linestyle="-", label="reward"
                )

                ax3 = ax2.twinx()
                ax3.set_ylabel("theta")
                ax3.spines["right"].set_edgecolor("red")
                ax3.yaxis.label.set_color("red")
                ax3.tick_params(axis="y", colors="red")
                line_3 = ax3.plot(theta_hist, "r--", label="theta")

                lines = line_2 + line_3
                labs = [label.get_label() for label in lines]
                ax3.legend(lines, labs, frameon=False, loc=4)
                fig.tight_layout()
                path_1 = (
                    self.path_prefix
                    + "/client{}_round{}_initdata{}/conv/convergence.png".format(
                        self.num_client,
                        self.num_round,
                        self.init_data_lower,
                    )
                )
                os.makedirs(os.path.dirname(path_1), exist_ok=True)
                fig.savefig(path_1, dpi=200)
                plt.close()

            else:
                max_diff_hist.append(max_diff)
                print("triumph2")
                return next_phi_list

        print("failure2")
        return next_phi_list

    def re_estimate_phi_theta(self, reward, theta):
        """
        函数4：给定R和theta，重新估计phi和delta
        """

        # 初始化phi_list
        phi_list = self.phi_list_init

        # 计算新的phi_list[T]
        for idx in range(self.fix_max_iter):
            increment_matrix, data_matrix, stale_matrix = self.estimate_D(
                phi_list, reward, theta
            )
            # print('******************************************************')
            # print('{}, {}, {}, {}'.format(idc, phi_list, self.reward, self.theta))
            # print('{}'.format(data_matrix))

            next_phi_list = np.sum(
                data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)),
                axis=0,
            )
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            # print('max_diff_phi:{}'.format(max_diff))

            if max_diff > self.fix_eps_3:
                phi_list = next_phi_list
                print("max_diff:{}".format(max_diff))
            else:
                print("triumph2")
                return next_phi_list, increment_matrix, data_matrix, stale_matrix

        print("failure2")
        return next_phi_list

    def estimate_direct_func(self, reward, data_matrix, stale_matrix):
        """
        函数5：给定reward, data_matrix, stale_matrix, 方便检查理论cost的各个部分
        """
        # 计算单列和
        data_list = []  # [T]
        for t in range(self.num_round):
            data = 0
            for k in range(self.num_client):
                data += data_matrix[k][t]
            data_list.append(data)

        res1 = 0
        res2 = 0
        res3 = 0
        res4 = 0
        res = 0
        for t in range(self.num_round):
            delta_sum = 0
            stale_sum = 0
            for k in range(self.num_client):
                delta_sum += data_matrix[k][t] * (self.delta_list[k] ** 2)
                stale_sum += data_matrix[k][t] * stale_matrix[k][t] * (self.sigma**2)

            # Omega不影响
            item_1 = (
                pow(self.kappa_1, self.num_round - 1 - t)
                * self.kappa_2
                * self.num_client
                * (self.psi**2)
                / data_list[t]
            )
            item_2 = (
                pow(self.kappa_1, self.num_round - 1 - t)
                * self.kappa_3
                * stale_sum
                / data_list[t]
            )
            item_3 = (
                pow(self.kappa_1, self.num_round - 1 - t)
                * self.kappa_4
                * delta_sum
                / data_list[t]
            )
            item = (1 - self.gamma) * (item_1 + item_2 + item_3) + self.gamma * reward
            res += item
            res1 += item_1
            res2 += item_2
            res3 += item_3
            res4 += self.gamma * reward
        return res, res1, res2, res3, res4

    def online_train(self):
        # 正式训练前定好一切
        path_2 = (
            self.path_prefix
            + "/client{}_round{}_initdata{}/pre/standard.npz".format(
                self.num_client, self.num_round, self.init_data_lower
            )
        )
        if os.path.exists(path_2) == False:
            phi_list = self.estimate_phi()  # [T]
            result = self.estimate_reward_theta(phi_list)  # [K, T]
            reward = np.array(result[0][0])
            theta = np.array(result[0][1])
            cost = np.array(result[1])
            os.makedirs(os.path.dirname(path_2), exist_ok=True)
            np.savez(path_2, phi=phi_list, reward=reward, theta=theta, cost=cost)
        print("has read")
        cache = np.load(path_2)
        reward = cache["reward"]
        theta = cache["theta"]

        # 尝试所有的变种
        gap_reward_list = []
        gap_theta_list = []
        # 变种对应的数据，画图2
        hist_increment_matrix = []
        hist_data_matrix = []
        hist_stale_matrix = []
        # 变种对应的cost理论值
        hist_res_list = []
        hist_res1_list = []
        hist_res2_list = []
        hist_res3_list = []
        hist_res4_list = []
        # 变种对应的loss/accuracy_list的最后一个，画图3
        hist_global_loss_list = []
        hist_accuracy_list = []
        # 变种对应的loss/accuracy_list的整个，画图4
        hist_loss_matrix = []
        hist_acc_matrix = []

        color = ["C0", "C2", "C3", "C4", "C5"]
        marker = ["^", "s", "o", "v", "D", "s", "+", "p", ","]
        linestyle = [":", "-", ":", ":"]
        sigma_dict = {1.25: 0.10, 1: 0.32, 0.75: 0.52, 0.475: 0.71, 0: 0.90}
        # enum = [[0, -theta + 0.3], [0, -theta + 0.52], [0, -theta + 0.70], [0, -theta + 0.90]]
        # enum = [[0, -theta + args['con_1']], [0, -theta + args['con_2']], [0, -theta + args['con_3']], [0, -theta + args['con_4']]]
        gaps = [[-reward, -theta + 1], [0, -theta + args["con_2"]]]
        for idx in range(len(gaps)):
            gap = gaps[idx]
            new_reward = reward + gap[0]
            new_theta = min(max(0.05, theta + gap[1]), 1)
            gap_reward_list.append(gap[0])
            gap_theta_list.append(gap[1])
            # print(reward, delta_com[0], new_reward)
            # print(theta, delta_com[1], new_theta)
            # exit(0)

            path_3 = (
                self.path_prefix
                + "/client{}_round{}_initdata{}/pre/variety/newreward{}_newtheta{}.npz".format(
                    self.num_client,
                    self.num_round,
                    self.init_data_lower,
                    str(round(new_reward, 2)),
                    str(round(new_theta, 2)),
                )
            )
            if os.path.exists(path_3) == False:
                var = self.re_estimate_phi_theta(new_reward, new_theta)
                phi_list = var[0]
                increment_matrix = var[1]
                data_matrix = var[2]
                stale_matrix = var[3]
                os.makedirs(os.path.dirname(path_3), exist_ok=True)
                np.savez(
                    path_3,
                    phi=phi_list,
                    increment=increment_matrix,
                    data=data_matrix,
                    stale=stale_matrix,
                )
            cache = np.load(path_3)
            phi_list = cache["phi"]
            increment_matrix = cache["increment"]
            data_matrix = cache["data"]
            stale_matrix = cache["stale"]
            # print(increment_matrix)
            # exit(0)

            hist_increment_matrix.append(np.mean(increment_matrix, axis=0))
            hist_data_matrix.append(np.mean(data_matrix, axis=0))
            hist_stale_matrix.append(np.mean(stale_matrix, axis=0))

            res, res1, res2, res3, res4 = self.estimate_direct_func(
                new_reward, data_matrix, stale_matrix
            )
            hist_res_list.append(res)
            hist_res1_list.append(res1)
            hist_res2_list.append(res2)
            hist_res3_list.append(res3)
            hist_res4_list.append(res4)

            if args["mode"] == "data":
                # 画图2，数据
                labels = []
                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    labels.append(label)
                    draw = hist_increment_matrix[n][::2]
                    draw = np.append(draw, hist_increment_matrix[n][-1])
                    plt.plot(
                        draw,
                        color=color[n],
                        marker=marker[n],
                        markersize=4,
                        linestyle=linestyle[n],
                        label=label,
                    )
                    plt.ylabel(
                        r"Averaged Data Collection Volume $\overline{\Delta}(t)$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 17,
                            "weight": "bold",
                        },
                    )
                    plt.xlabel(
                        r"Round $t$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 18,
                            "weight": "bold",
                        },
                    )
                    plt.yticks(fontproperties="Times New Roman", size=10)
                    plt.xticks(
                        range(len(draw)),
                        [i for i in range(0, self.num_round + 1, 2)],
                        fontproperties="Times New Roman",
                        size=10,
                    )
                    plt.legend(
                        frameon=False,
                        prop={
                            "family": "Times New Roman",
                            "size": 10,
                            "weight": "bold",
                        },
                    )
                    path_4 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/data/delta.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                        )
                    )
                    os.makedirs(os.path.dirname(path_4), exist_ok=True)
                    plt.savefig(
                        path_4,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

                labels = []
                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    labels.append(label)
                    draw = hist_data_matrix[n][::2]
                    draw = np.append(draw, hist_data_matrix[n][-1])
                    plt.plot(
                        draw,
                        color=color[n],
                        marker=marker[n],
                        markersize=4,
                        linestyle=linestyle[n],
                        label=label,
                    )
                    plt.ylabel(
                        r"Averaged Buffered Data Volume $\overline{D}(t)$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 18,
                            "weight": "bold",
                        },
                    )
                    plt.xlabel(
                        r"Round $t$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 18,
                            "weight": "bold",
                        },
                    )
                    plt.yticks(fontproperties="Times New Roman", size=10)
                    plt.xticks(
                        range(len(draw)),
                        [i for i in range(0, self.num_round + 1, 2)],
                        fontproperties="Times New Roman",
                        size=10,
                    )
                    plt.ylim(0, 1000)
                    plt.legend(
                        frameon=False,
                        prop={
                            "family": "Times New Roman",
                            "size": 10,
                            "weight": "bold",
                        },
                    )
                    path_5 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/data/data.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                        )
                    )
                    os.makedirs(os.path.dirname(path_5), exist_ok=True)
                    plt.savefig(
                        path_5,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

                labels = []
                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    labels.append(label)
                    draw = hist_stale_matrix[n][::2]
                    draw = np.append(draw, hist_stale_matrix[n][-1])
                    plt.plot(
                        draw,
                        color=color[n],
                        marker=marker[n],
                        markersize=4,
                        linestyle=linestyle[n],
                        label=label,
                    )
                    plt.ylabel(
                        r"Averaged Staleness of Data $\overline{S}(t)$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 18,
                            "weight": "bold",
                        },
                    )
                    plt.xlabel(
                        r"Round $t$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 18,
                            "weight": "bold",
                        },
                    )
                    plt.yticks(fontproperties="Times New Roman", size=10)
                    plt.xticks(
                        range(len(draw)),
                        [i for i in range(0, self.num_round + 1, 2)],
                        fontproperties="Times New Roman",
                        size=10,
                    )
                    plt.ylim(0, 16)
                    plt.legend(
                        frameon=False,
                        prop={
                            "family": "Times New Roman",
                            "size": 10,
                            "weight": "bold",
                        },
                    )
                    path_6 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/data/stale.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                        )
                    )
                    os.makedirs(os.path.dirname(path_6), exist_ok=True)
                    plt.savefig(
                        path_6,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

                flag = 0
                width = 0.17
                x = range(len(gap_reward_list))
                ticks = [
                    r"$(\theta={:.2f})$".format(min(max(0.05, theta + gap_theta), 1))
                    for gap_theta in gap_theta_list
                ]
                for n in range(idx + 1):
                    plt.bar(
                        np.array(range(len(hist_res_list))) - 1.5 * width,
                        hist_res_list,
                        width=width,
                        label="total",
                    )
                    plt.bar(
                        np.array(range(len(hist_res1_list))) - 0.5 * width,
                        hist_res1_list,
                        width=width,
                        label="datasize",
                    )
                    plt.bar(
                        np.array(range(len(hist_res2_list))) + 0.5 * width,
                        hist_res2_list,
                        width=width,
                        label="age",
                    )
                    plt.bar(
                        np.array(range(len(hist_res4_list))) + 1.5 * width,
                        hist_res4_list,
                        width=width,
                        label="reward",
                    )
                    plt.xticks(
                        x, ticks[: len(x)], fontproperties="Times New Roman", size=10
                    )
                    plt.ylabel(
                        r"Theoretic Cost $U$",
                        fontdict={
                            "family": "Times New Roman",
                            "size": 14,
                            "weight": "bold",
                        },
                    )
                    if flag == 0:
                        plt.legend(frameon=False)
                        flag = 1
                    path_7 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/data/cost.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                        )
                    )
                    os.makedirs(os.path.dirname(path_7), exist_ok=True)
                    plt.savefig(
                        path_7,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

            if args["mode"] == "train":
                # 初始化数据和网络
                self.init_data_net()
                global_parameter = {}
                for key, var in self.net.state_dict().items():
                    global_parameter[key] = var.clone()
                # 计算聚合权重
                rate_matrix = np.stack(
                    [
                        data_matrix[:, t] / sum(data_matrix[:, t])
                        for t in range(self.num_round)
                    ]
                )
                rate_matrix = torch.from_numpy(rate_matrix).T
                # 记录loss与acc
                global_loss_list = []
                accuracy_list = []
                # 训练
                for t in tqdm(range(self.num_round)):
                    next_global_parameter = {}
                    global_loss = 0
                    for k in range(self.num_client):
                        result = self.client_group.clients[k].local_update_avg(
                            idx,
                            t,
                            k,
                            self.sigma,
                            self.num_epoch,
                            self.batch_size,
                            global_parameter,
                            new_theta,  # 妈的是new的不是旧的
                            increment_matrix[k][t],
                            data_matrix[k][t],
                            args["poison_sigma"],
                        )
                        local_parameter = result[0]
                        for item in local_parameter.items():
                            if item[0] not in next_global_parameter.keys():
                                next_global_parameter[item[0]] = (
                                    rate_matrix[k][t] * item[1].clone()
                                )
                            else:
                                next_global_parameter[item[0]] += (
                                    rate_matrix[k][t] * item[1].clone()
                                )
                        # print(
                        #     local_parameter["conv_block.0.weight"].data_ptr()
                        #     == next_global_parameter["conv_block.0.weight"].data_ptr()
                        # )
                        # exit(0)
                        # 不会污染第一个本地模型

                        local_loss = result[1]
                        global_loss += rate_matrix[k][t] * local_loss

                    # 求global_parameters和global_loss_list
                    global_parameter = next_global_parameter
                    global_loss_list.append(global_loss)

                    # 验证
                    if t % self.eval_freq == 0 or t == self.num_round - 1:
                        correct = 0
                        total = 0
                        self.net.load_state_dict(global_parameter)
                        with torch.no_grad():
                            # 固定哦
                            test_dataloader = DataLoader(
                                ConcatDataset(self.test_data_list),
                                batch_size=1024,
                                shuffle=False,
                            )
                            for batch in test_dataloader:
                                data, label = batch
                                # print(label)
                                data = data.to(self.dev)
                                label = label.to(self.dev)
                                pred = self.net(data)  # [batch_size， 10]，输出的是概率
                                pred = torch.argmax(pred, dim=1)
                                correct += (pred == label).sum().item()
                                total += label.shape[0]
                        acc = correct / total
                        accuracy_list.append(acc)

                # 画图3，真实的cost
                hist_global_loss_list.append(global_loss_list[-1])
                hist_accuracy_list.append(accuracy_list[-1])

                width = 0.8
                x = range(len(gap_reward_list))
                labels = []
                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    labels.append(label)
                cost_1_list = (1 - self.new_gamma) * (1 - np.array(hist_accuracy_list))
                cost_2_list = self.new_gamma * (np.array(gap_reward_list) + reward)
                plt.bar(
                    x,
                    cost_1_list,
                    width=0.7 * width,
                    hatch="/",
                    color="C0",
                    label="loss",
                )
                plt.bar(
                    x,
                    cost_2_list,
                    width=0.7 * width,
                    hatch="\\",
                    color="C2",
                    label="reward",
                    bottom=cost_1_list,
                )
                plt.ylabel(
                    r"Realistic Cost of Server $U$",
                    fontproperties="Times New Roman",
                    size=20,
                )
                plt.xlabel(r"Server Strategy", size=20)
                plt.yticks(fontproperties="Times New Roman", size=17)
                plt.xticks(x, labels, fontproperties="Times New Roman", size=17)
                # plt.ylim(0, 0.5)
                plt.legend(
                    frameon=False,
                    prop={"family": "Times New Roman", "size": 17, "weight": "bold"},
                )
                path_8 = (
                    self.path_prefix
                    + "/client{}_round{}_initdata{}/{}_{}_{}/cost/d{}_e{}_i{}.png".format(
                        self.num_client,
                        self.num_round,
                        self.init_data_lower,
                        self.dataset_name,
                        self.net_name,
                        self.alg_name,
                        args["dirichlet"],
                        args["num_epoch"],
                        args["init_num_class"],
                    )
                )
                os.makedirs(os.path.dirname(path_8), exist_ok=True)
                plt.savefig(
                    path_8,
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()

                # 画图4，真实的loss和accuracy
                hist_loss_matrix.append(global_loss_list)
                hist_acc_matrix.append(accuracy_list)
                tmp1 = int(self.num_round / self.eval_freq) + 1
                tmp2 = self.num_round + self.eval_freq
                tmp3 = self.eval_freq

                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    draw = hist_loss_matrix[n][::5]
                    draw = np.append(draw, hist_loss_matrix[n][-1])
                    plt.plot(
                        draw,
                        color=color[n],
                        marker=marker[n],
                        linestyle=linestyle[n],
                        label=label,
                    )
                    plt.ylabel("Loss", fontproperties="Times New Roman", size=20)
                    plt.xlabel("Round $t$", fontproperties="Times New Roman", size=20)
                    # plt.ylim(0.15, 0.95)
                    plt.yticks(fontproperties="Times New Roman", size=17)
                    plt.xticks(
                        range(0, tmp1),
                        range(0, tmp2, tmp3),
                        fontproperties="Times New Roman",
                        size=17,
                    )
                    plt.legend(
                        frameon=False,
                        prop={
                            "family": "Times New Roman",
                            "size": 17,
                            "weight": "bold",
                        },
                    )
                    path_9 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/{}_{}_{}/loss/d{}_e{}_i{}.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                            self.dataset_name,
                            self.net_name,
                            self.alg_name,
                            args["dirichlet"],
                            args["num_epoch"],
                            args["init_num_class"],
                        )
                    )
                    os.makedirs(os.path.dirname(path_9), exist_ok=True)
                    plt.savefig(
                        path_9,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

                count = 1
                for n in range(idx + 1):
                    tmp_theta = min(max(0.05, theta + gap_theta_list[n]), 1)
                    if tmp_theta == 0.5:
                        label = r"$(R, \theta)^*$"
                    else:
                        label = r"$(R, \theta)^{}$".format(count)
                        count += 1
                    # plt.plot(hist_acc_matrix[n], color=color[n], marker=marker[n], linestyle=linestyle[n], label=label)
                    plt.plot(
                        hist_acc_matrix[n],
                        color=color[n],
                        linestyle=linestyle[n],
                        label=label,
                    )
                    plt.ylabel("Accuracy", fontproperties="Times New Roman", size=20)
                    plt.xlabel("Round $t$", fontproperties="Times New Roman", size=20)
                    plt.ylim(0.15, 0.95)
                    plt.yticks(fontproperties="Times New Roman", size=17)
                    plt.xticks(
                        range(0, tmp1),
                        range(0, tmp2, tmp3),
                        fontproperties="Times New Roman",
                        size=17,
                    )
                    plt.legend(
                        frameon=False,
                        prop={
                            "family": "Times New Roman",
                            "size": 17,
                            "weight": "bold",
                        },
                    )
                    path_10 = (
                        self.path_prefix
                        + "/client{}_round{}_initdata{}/{}_{}_{}/acc/d{}_e{}_i{}.png".format(
                            self.num_client,
                            self.num_round,
                            self.init_data_lower,
                            self.dataset_name,
                            self.net_name,
                            self.alg_name,
                            args["dirichlet"],
                            args["num_epoch"],
                            args["init_num_class"],
                        )
                    )
                    os.makedirs(os.path.dirname(path_10), exist_ok=True)
                    plt.savefig(
                        path_10,
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()


def parser_args():
    parser = argparse.ArgumentParser()

    # 只列出你想通过命令行覆盖的参数
    parser.add_argument("--dirichlet", type=float, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--init_num_class", type=int, default=None)

    cli_args = vars(parser.parse_args())
    dirichlet = cli_args["dirichlet"]
    num_epoch = cli_args["num_epoch"]
    init_num_class = cli_args["init_num_class"]

    # 加载默认参数
    with open("../config/args.yaml") as f:
        default_args = yaml.safe_load(f)

    # 覆盖默认参数
    default_args["dirichlet"] = dirichlet
    default_args["num_epoch"] = num_epoch
    default_args["init_num_class"] = init_num_class

    return default_args


if __name__ == "__main__":
    # 设置随机种子
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args = parser_args()
    print(
        f"Running FedStream with: Dir={args['dirichlet']}, Epoch={args['num_epoch']}, Init_Class={args['init_num_class']}"
    )

    server = Server(args)
    server.online_train()
