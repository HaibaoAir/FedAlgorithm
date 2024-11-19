        # # 定义loss function
        # self.criterion = F.cross_entropy # 交叉熵：softmax + NLLLoss 参考知乎
        
        # # 定义optimizer
        # self.eta = args['learning_rate']
        # self.optim = torch.optim.SGD(self.net.parameters(), self.eta)
    
    # def estimate_D(self, phi_list, reward_list):
    #     # 初始化data_matrix[K,T+1]
    #     data_init = [random.randint(50, 100) for _ in range(self.num_client)]
    #     # print('data_init:{}'.format(data_init))
    #     data_matrix = []
    #     for k in range(self.num_client):
    #         data_list = [data_init[k]]
    #         for t in range(1, self.num_round + 1):
    #             data = random.random() * 60
    #             data_list.append(data)
    #         data_matrix.append(data_list)
        
    #     for idc in range(self.fix_max_iter):
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
    #             next_data_list = [data_init[k]]
    #             for t in range(1, self.num_round + 1):
    #                 next_data = self.theta * next_data_list[t - 1] + increment_matrix[k][t - 1]
    #                 next_data_list.append(next_data)
    #             next_data_matrix.append(next_data_list)
            
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
    #             # print('YES:{}'.format(idc))
    #             # for k in range(self.num_client):
    #             #     print(increment_matrix[k])
    #             for k in range(self.num_client):
    #                 print(next_data_matrix[k])
    #             print('-----------------------------')
    #             return next_data_matrix
    #     return next_data_matrix
            

    # def estimate_R(self, phi_list):
        
    #     def episode(reward_list):
    #         data_matrix = self.estimate_D(phi_list, reward_list)
    #         self.init_params(args)
    #         res = self.train(reward_list, data_matrix)
    #         # print('##################################')
    #         # exit(0)
    #         res = res.cpu()
    #         res = res.detach().numpy()
    #         return res
        
    #     global count
    #     count = 0
    #     pso =  PSO(func=episode,
    #                dim=self.num_round + 1,
    #                pop=self.pop,
    #                max_iter=self.pso_max_iter,
    #                lb=[0 for _ in range(self.num_round + 1)],
    #                ub=[self.ub for _ in range(self.num_round + 1)],
    #                eps=self.pso_eps)
    #     pso.run()
    #     return pso.gbest_x, pso.gbest_y, pso.gbest_y_hist
    
    # def train(self, reward_list, data_matrix):
    #     global count
    #     count = count + 1
    #     print('count:{}'.format(count))
    #     self.global_parameter = {}
    #     for key, var in self.net.state_dict().items():
    #         self.global_parameter[key] = var.clone()
            
    #     data_t_list = []
    #     for t in range(self.num_round + 1):
    #         data_t = sum(data_matrix[k][t] for k in range(self.num_client))
    #         data_t_list.append(data_t)
        
    #     # 训练
    #     res = 0
    #     for t in tqdm(range(self.num_round + 1)):
    #         next_global_parameter = {}
    #         old_global_loss = 0
    #         new_global_loss = 0
    #         new_global_grad_list = []
    #         # print('server:')
    #         # print(id(self.global_parameter))
    #         # print(self.global_parameter['fc2.bias'])
    #         for k in range(self.num_client):
    #             item = self.client_group.clients[k].local_update(t,
    #                                                             k,
    #                                                             self.num_epoch,
    #                                                             self.batch_size,
    #                                                             self.global_parameter,
    #                                                             self.theta,
    #                                                             data_matrix[k][t])
                
    #             local_parameter = item[0]
    #             old_local_loss = item[1]
    #             new_local_loss = item[2]
    #             new_local_grad = item[3]
                
    #             new_rate = data_matrix[k][t] / data_t_list[t]
    #             new_global_loss += new_rate * new_local_loss
    #             new_global_grad_list.append(new_rate * new_local_grad)   

    #             for item in local_parameter.items():
    #                 if item[0] not in next_global_parameter.keys():
    #                     next_global_parameter[item[0]] = new_rate * item[1]
    #                 else:
    #                     next_global_parameter[item[0]] += new_rate * item[1]
                
    #             if t != 0:
    #                 old_rate = data_matrix[k][t-1] / data_t_list[t-1]
    #                 old_global_loss += old_rate * old_local_loss 
                
    #         # 求global_parameters和5个参数
    #         self.global_parameter = next_global_parameter
    #         omega = new_global_loss - old_global_loss            
    #         delta = 0
    #         new_global_grad = sum(new_global_grad_list)
    #         new_rate = data_matrix[k][t] / data_t_list[t]
    #         for item in new_global_grad_list:
    #             delta += new_rate * torch.sqrt(torch.sum(torch.square(item - new_global_grad))) # 用delta代替Upsilon
            
    #         # 总计算
    #         Theta = 1
    #         kappa1 = 1
    #         kappa2 = 1
    #         item1 = kappa1 * delta ** 2 # delta替代也是个坑
    #         item2 = sum(1 / epsilon ** 2 for epsilon in self.epsilon_list) / data_t_list[t] ** 2
    #         item3 = Theta * omega
    #         item4 = pow(Theta, self.num_round - 1 - t)
    #         item5 = pow(self.gamma, t) * reward_list[t]
    #         if t == self.num_round:
    #             item1 = 0
    #             item2 = 0
    #         if t == 0:
    #             item3 = 0
    #         res += item4 * (item1 + item2 + item3) + item5
    #         print('t={}, lambda={}, omega={}, reward={}, res={}'.format(t, item1 + item2, item3, item5, res))
    #     return res
    
    # def online_estimate_phi(self):
    #     # 初始化phi_list[T+1]
    #     phi_list = []
    #     for t in range(self.num_round + 1):
    #         phi = random.random() * 60
    #         phi_list.append(phi)
        
    #     # 计算新的phi_list[T + 1]
    #     for _ in range(self.fix_max_iter):
    #         self.init_params(args)
    #         avg_Upsilon_matrix, pre_increment_matrix = self.online_train(phi_list)
    #         next_phi_list = self.epsilon_list * (pre_increment_matrix / avg_Upsilon_matrix)
        
    #         # 判断收敛
    #         if torch.max(abs(next_phi_list - phi_list)) > self.eps:
    #             phi_list = next_phi_list
    #         else:
    #             return next_phi_list  