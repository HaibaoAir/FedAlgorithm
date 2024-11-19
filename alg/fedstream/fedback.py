    # def pretrain(self, phi_list, reward_list):
        
    #     # 初始化data_matrix[K,T+1]
    #     data_init = 20
    #     data_matrix = []
    #     for k in range(self.num_client):
    #         data_list = [data_init]
    #         for t in range(1, self.num_round + 1):
    #             data = random.random() * 60
    #             data_list.append(data)
    #         data_matrix.append(data_list)
        
    #     for idc in range(10):
    #         # 计算phi_list[T + 1]
    #         phi_list = []
    #         for t in range(self.num_round + 1):
    #             phi = 0
    #             for k in range(self.num_client):
    #                 phi += self.epsilon_list[k] * self.Upsilon_matrix[k][t] * data_matrix[k][t]
    #             phi_list.append(phi)
            
    #         # 计算增量矩阵[K,T]
    #         increment_matrix = []
    #         for k in range(self.num_client):
    #             increment_list = []
    #             for t in range(self.num_round):
    #                 item = 0
    #                 for tau in range(t + 1, self.num_round + 1):
    #                     item1 = pow(self.theta, tau - t - 1)
    #                     item2 = self.epsilon_list[k] * self.Upsilon_matrix[k][tau] * reward_list[tau] / phi_list[tau]
    #                     item3 = 2 * self.beta_list[k] * data_matrix[k][tau]
    #                     item += item1 * (item2 - item3)
    #                 increment = 1 / (2 * self.alpha_list[k]) * item
    #                 increment_list.append(increment)
    #             increment_matrix.append(increment_list)
                
    #         # 新的数据矩阵[K, T+1]
    #         next_data_matrix = []
    #         for k in range(self.num_client):
    #             next_data_list = [data_init]
    #             for t in range(1, self.num_round + 1):
    #                 next_data = self.theta * next_data_list[t - 1] + increment_matrix[k][t - 1]
    #                 next_data_list.append(next_data)
    #             next_data_matrix.append(next_data_list)
            
    #         # print('iter{}:'.format(idc))
    #         # print('data_matrix:')
    #         # for item in data_matrix:
    #         #     print(item)
    #         # print('increment_matrix:')
    #         # for item in increment_matrix:
    #         #     print(item)
    #         # print('------------------------')
    #         # draw.append(data_matrix[0][1])

    #         # 判断收敛
    #         flag = 0
    #         for k in range(self.num_client):
    #             for t in range(1, self.num_round + 1):
    #                 if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.eps:
    #                     flag = 1
    #                     break
    #             if flag == 1:
    #                 break
    #         if flag == 1:
    #             data_matrix = next_data_matrix
    #         else:
    #             return next_data_matrix
    #     plt.plot(draw)
    #     plt.savefig('../../logs/disconvergence.jpg')
    #     exit(0)