import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import argparse
import yaml

sys.path.append("../")
from config.config import args_parser
from FedAvg import Server as FedAvg_Server
from FedProx import Server as FedProx_Server
from FedNova import Server as FedNova_Server
from FedDyn import Server as FedDyn_Server


class FedStream(object):
    def __init__(self, args):
        # 初始化data_matrix[K,T]
        self.num_client = args.num_client
        self.num_round = args.num_round
        self.eval_freq = args.eval_freq
        self.path_prefix = args.path_prefix
        self.init_data_lb = args.init_data_lb
        self.init_data_ub = args.init_data_ub
        self.alg_name = args.alg_name
        self.net_name = args.net_name
        self.dataset_name = args.dataset_name
        self.dirichlet = args.dirichlet
        self.init_num_class = args.init_num_class
        self.num_epoch = args.num_epoch

        self.data_origin_init = [
            random.randint(self.init_data_lb, self.init_data_ub)
            for _ in range(self.num_client)
        ]
        self.data_matrix_init = []
        for k in range(self.num_client):
            data_list = [self.data_origin_init[k]]
            for t in range(1, self.num_round):
                data = random.randint(50, 100)
                data_list.append(data)
            self.data_matrix_init.append(data_list)

        self.phi_list_init = []
        for t in range(self.num_round):
            phi = random.random() * 60
            self.phi_list_init.append(phi)

        self.delta_list = np.array([args.delta] * args.num_client)  # [K]
        self.psi = args.psi
        self.alpha_list = [args.alpha] * self.num_client  # 收集数据的价格
        self.beta_list = [
            args.beta
        ] * self.num_client  # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.sigma = args.sigma

        self.kappa_1 = args.kappa_1
        self.kappa_2 = args.kappa_2
        self.kappa_3 = args.kappa_3
        self.kappa_4 = args.kappa_4
        self.gamma = args.gamma
        self.new_gamma = args.new_gamma

        self.reward_lb = args.reward_lb  # 反正不看过程只看结果，原来1-100得50
        self.reward_ub = args.reward_ub
        self.theta_lb = args.theta_lb
        self.theta_ub = args.theta_ub
        self.n_calls = args.n_calls
        self.base_estimator = args.base_estimator
        self.noise = args.noise
        self.acq_func = args.acq_func
        self.xi = args.xi
        self.random_state = args.random_state

        self.fix_eps_1 = args.fix_eps_1
        self.fix_eps_2 = args.fix_eps_2
        self.fix_eps_3 = args.fix_eps_3
        self.fix_max_iter = args.fix_max_iter

        # 选择算法
        if self.alg_name == "fedavg":
            self.alg = FedAvg_Server(args)
        elif self.alg_name == "fedprox":
            self.alg = FedProx_Server(args)
        elif self.alg_name == "feddyn":
            self.alg = FedDyn_Server(args)
        elif self.alg_name == "fednova":
            self.alg = FedNova_Server(args)
        else:
            raise ValueError("Invalid algorithm name")

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
                        self.init_data_lb,
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

    def re_estimate_cost_part(self, reward, data_matrix, stale_matrix):
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

    def draw_data(self, scenarios, matrix, topic):
        color = ["C2", "C0", "C3", "C4", "C5"]
        marker = ["s", "^", "o", "v", "D", "s", "+", "p", ","]
        linestyle = ["-", ":", ":", ":"]

        for n in range(len(matrix)):
            draw = matrix[n][:: self.eval_freq]
            draw = np.append(draw, matrix[n][-1])
            plt.plot(
                draw,
                color=color[n],
                marker=marker[n],
                markersize=4,
                linestyle=linestyle[n],
                label=r"$(R={},\theta={})$".format(scenarios[n][0], scenarios[n][1]),
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
            range(0, self.num_round + self.eval_freq, self.eval_freq),
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
        path_4 = self.path_prefix + "/client{}_round{}_initdata{}/data/{}.png".format(
            self.num_client,
            self.num_round,
            self.init_data_lb,
            topic,
        )
        os.makedirs(os.path.dirname(path_4), exist_ok=True)
        plt.savefig(
            path_4,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def draw_cost(self, scenarios, vector):

        width = 0.8
        x = range(len(vector))
        cost_1_list = (1 - self.new_gamma) * (1 - np.array(vector))
        cost_2_list = self.new_gamma * (np.array(scenarios)[:, 0])
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
        plt.xticks(x, scenarios[: len(x)], fontproperties="Times New Roman", size=17)
        # plt.ylim(0, 0.5)
        plt.legend(
            frameon=False,
            prop={
                "family": "Times New Roman",
                "size": 17,
                "weight": "bold",
            },
        )
        path_5 = (
            self.path_prefix
            + "/client{}_round{}_initdata{}/{}_{}_{}/cost/d{}_c{}_e{}.png".format(
                self.num_client,
                self.num_round,
                self.init_data_lb,
                self.dataset_name,
                self.net_name,
                self.alg_name,
                self.dirichlet,
                self.init_num_class,
                self.num_epoch,
            )
        )
        os.makedirs(os.path.dirname(path_5), exist_ok=True)
        plt.savefig(
            path_5,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def draw_loss_acc(self, scenarios, hist_loss_matrix, hist_acc_matrix):
        color = ["C2", "C0", "C3", "C4", "C5"]
        marker = ["s", "^", "o", "v", "D", "s", "+", "p", ","]
        linestyle = ["-", ":", ":", ":"]

        for n in range(len(hist_loss_matrix)):
            draw = hist_loss_matrix[n][:: self.eval_freq]
            draw = np.append(draw, hist_loss_matrix[n][-1])
            plt.plot(
                draw,
                color=color[n],
                marker=marker[n],
                linestyle=linestyle[n],
                label=r"$(R={},\theta={})$".format(scenarios[n][0], scenarios[n][1]),
            )
            plt.ylabel("Loss", fontproperties="Times New Roman", size=20)
            plt.xlabel("Round $t$", fontproperties="Times New Roman", size=20)
            plt.ylim(0.15, 10)
            plt.yticks(fontproperties="Times New Roman", size=17)
            plt.xticks(
                range(len(draw)),
                range(0, self.num_round + self.eval_freq, self.eval_freq),
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
            path_6 = (
                self.path_prefix
                + "/client{}_round{}_initdata{}/{}_{}_{}/loss/d{}_c{}_e{}".format(
                    self.num_client,
                    self.num_round,
                    self.init_data_lb,
                    self.dataset_name,
                    self.net_name,
                    self.alg_name,
                    self.dirichlet,
                    self.init_num_class,
                    self.num_epoch,
                )
            )
            os.makedirs(os.path.dirname(path_6), exist_ok=True)
            plt.savefig(
                path_6 + ".png",
                dpi=200,
                bbox_inches="tight",
            )
            with open(path_6 + ".txt", "w") as f:
                f.write(str(np.array(hist_loss_matrix)))
        plt.close()

        for n in range(len(hist_acc_matrix)):
            draw = hist_acc_matrix[n]
            plt.plot(
                draw,
                color=color[n],
                marker=marker[n],
                linestyle=linestyle[n],
                label=r"$(R={},\theta={})$".format(scenarios[n][0], scenarios[n][1]),
            )
            plt.ylabel(r"Accuracy", fontproperties="Times New Roman", size=20)
            plt.xlabel(r"Round $t$", fontproperties="Times New Roman", size=20)
            plt.ylim(0.15, 0.95)
            plt.yticks(fontproperties="Times New Roman", size=17)
            plt.xticks(
                range(len(draw)),
                range(0, self.num_round + self.eval_freq, self.eval_freq),
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
            path_7 = (
                self.path_prefix
                + "/client{}_round{}_initdata{}/{}_{}_{}/acc/d{}_c{}_e{}".format(
                    self.num_client,
                    self.num_round,
                    self.init_data_lb,
                    self.dataset_name,
                    self.net_name,
                    self.alg_name,
                    self.dirichlet,
                    self.init_num_class,
                    self.num_epoch,
                )
            )
            os.makedirs(os.path.dirname(path_7), exist_ok=True)
            plt.savefig(
                path_7 + ".png",
                dpi=200,
                bbox_inches="tight",
            )
            with open(path_7 + ".txt", "w") as f:
                f.write(str(np.array(hist_acc_matrix)))
        plt.close()

    def online_train(self):
        # 正式训练前定好一切
        path_2 = (
            self.path_prefix
            + "/client{}_round{}_initdata{}/pre/standard.npz".format(
                self.num_client, self.num_round, self.init_data_lb
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
        reward = np.round(cache["reward"], 2)
        theta = np.round(cache["theta"], 2)

        # 尝试所有的变种
        # 变种对应的数据，画图2
        hist_increment_matrix = []
        hist_data_matrix = []
        hist_stale_matrix = []
        # 变种对应的cost理论值，画图2
        hist_res_list = []
        hist_res1_list = []
        hist_res2_list = []
        hist_res3_list = []
        hist_res4_list = []
        # 画图3
        hist_loss_list = []
        hist_acc_list = []
        # 画图4
        hist_loss_matrix = []
        hist_acc_matrix = []

        # scenarios = [[reward, theta], [0.0, 1.0]]
        scenarios = [[reward, 0.5], [reward, 0.8]]
        for idx in range(len(scenarios)):
            new_reward = scenarios[idx][0]
            new_theta = min(max(0.05, scenarios[idx][1]), 1)
            path_3 = (
                self.path_prefix
                + "/client{}_round{}_initdata{}/pre/variety/newreward{}_newtheta{}.npz".format(
                    self.num_client,
                    self.num_round,
                    self.init_data_lb,
                    str(new_reward),
                    str(new_theta),
                )
            )
            if os.path.exists(path_3) == False:
                var = self.re_estimate_phi_theta(new_reward, new_theta)
                new_phi_list = var[0]
                new_increment_matrix = var[1]
                new_data_matrix = var[2]
                new_stale_matrix = var[3]
                os.makedirs(os.path.dirname(path_3), exist_ok=True)
                np.savez(
                    path_3,
                    phi=new_phi_list,
                    increment=new_increment_matrix,
                    data=new_data_matrix,
                    stale=new_stale_matrix,
                )
            cache = np.load(path_3)
            new_phi_list = cache["phi"]
            new_increment_matrix = cache["increment"]
            new_data_matrix = cache["data"]
            new_stale_matrix = cache["stale"]
            #################
            new_increment_matrix = new_increment_matrix * 10
            for k in range(self.num_client):
                for t in range(1, self.num_round):
                    new_data_matrix[k][t] = (
                        new_theta * new_data_matrix[k][t - 1]
                        + new_increment_matrix[k][t]
                    )
            #################
            hist_increment_matrix.append(np.mean(new_increment_matrix, axis=0))
            hist_data_matrix.append(np.mean(new_data_matrix, axis=0))
            hist_stale_matrix.append(np.mean(new_stale_matrix, axis=0))

            res, res1, res2, res3, res4 = self.re_estimate_cost_part(
                new_reward, new_data_matrix, new_stale_matrix
            )
            hist_res_list.append(res)
            hist_res1_list.append(res1)
            hist_res2_list.append(res2)
            hist_res3_list.append(res3)
            hist_res4_list.append(res4)

            # 训练
            global_loss_list, accuracy_list = self.alg.run(
                new_theta,  # 妈的是new的不是旧的
                new_data_matrix,
            )
            hist_loss_list.append(global_loss_list[-1])
            hist_acc_list.append(accuracy_list[-1])
            hist_loss_matrix.append(global_loss_list)
            hist_acc_matrix.append(accuracy_list)

            # 画增量
            self.draw_data(scenarios, hist_increment_matrix, "delta")
            # 画数据
            self.draw_data(scenarios, hist_data_matrix, "data")
            # 画stale
            self.draw_data(scenarios, hist_stale_matrix, "stale")
            # # 画真实成本
            # self.draw_cost(scenarios, hist_acc_list)
            # 画精确度
            self.draw_loss_acc(scenarios, hist_loss_matrix, hist_acc_matrix)


if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    # 设置随机种子
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args = args_parser()
    print(
        f"Running FedStream with: Dir={args.dirichlet}, Init_Class={args.init_num_class}, Epoch={args.num_epoch}"
    )

    alg = FedStream(args)
    alg.online_train()
