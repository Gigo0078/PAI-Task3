base_net1 = NeuralNetwork(input_dim = self.state_dim + self.action_dim, 
                                  output_dim = 1, 
                                  hidden_size = 256,
                                  hidden_layers = 2,
                                  activation="relu").to(self.device) #self.critic_Q1.NN_critic #self.critic_Q1.NN_critic
        
        base_net2 = NeuralNetwork(input_dim = self.state_dim + self.action_dim, 
                                  output_dim = 1, 
                                  hidden_size = 256,
                                  hidden_layers = 2,
                                  activation="relu").to(self.device) #self.critic_Q2.NN_critic

        base_net1.load_state_dict(copy.deepcopy(self.critic_Q1.NN_critic.state_dict()))
        base_net1.to(self.device)
        base_net2.load_state_dict(copy.deepcopy(self.critic_Q2.NN_critic.state_dict()))
        base_net2.to(self.device)