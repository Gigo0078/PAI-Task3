import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
from scipy.stats import norm
import copy
import random

torch.manual_seed(5)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
            }
        self.activation = self.activations[activation]
        self.input = nn.Linear(self.input_dim, self.hidden_size)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.hidden_layers)])
        self.putput = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        #pass
        s = self.input(s)
        s = self.activation(s)
        for i in range(0,self.hidden_layers):
            s = self.linears[i](s)
            s = self.activation(s)
        s = self.putput(s)
        #s = self.activation(s)
        return s

    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        #pass
        self.NN_actor = NeuralNetwork(input_dim=self.state_dim,
                                       output_dim=2*self.action_dim,
                                         hidden_size=self.hidden_size, 
                                         hidden_layers=self.hidden_layers,
                                         activation="relu")
        self.NN_actor.to(self.device)
        self.optimizer= optim.Adam(self.NN_actor.parameters(),lr = self.actor_lr)
        #self.temperature = TrainableParameter(init_param=0.005, lr_param=0.1, train_param=True)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool = False) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        #print("input state is", state)
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.


        #########   OLD VERSION    #########
        
        # state = torch.tensor(state)

        # mean, std = torch.chunk(self.NN_actor(state), 2, dim=-1)#.to(self.device)

        # std = torch.tensor(std)
        # mean = torch.tensor(mean)
        
        # log_std = self.clamp_log_std(torch.log(std))   #The log of the standard deviation must be clamped not the standard deviation
        # std = torch.exp(log_std)
        # dist = torch.distributions.normal.Normal(mean, std)

        # if deterministic == False:  #We aren't sure about the placement of the clamping, as it makes a difference for the probability, what its std is
        #     action = dist.rsample() #rsample includes the reparametrization trick
        #     action = torch.tanh(action)
        # else:
        #     action = mean

                    
        # log_prob = dist.log_prob(action)
            
        # action = action.reshape((self.action_dim,))
        # log_prob = torch.tensor(log_prob.reshape((self.action_dim,)))

        # assert action.shape == (self.action_dim, ) and \
        #     log_prob.shape == (self.action_dim, ), 'Incorrect shape for action or log_prob.'
        
        # return action, log_prob

        ############# TEST VERSION #############

        outputs = self.NN_actor(state)

        if state.shape == (3,): # one state
            outputs = [outputs]

        #print("outputs", outputs)

        actions, log_probs = [], []

        for out in outputs:

            #print("out", out)

            mean, std = out # need to check if can unpack

            std = torch.tensor(torch.abs(std))
            mean = torch.tensor(mean)

            log_std = self.clamp_log_std(torch.log(std))   #The log of the standard deviation must be clamped not the standard deviation
            std = torch.exp(log_std)

            #print("mean", mean)
            #print("std", std)

            dist = torch.distributions.normal.Normal(mean, std)

            if deterministic == False:  #We aren't sure about the placement of the clamping, as it makes a difference for the probability, what its std is
        
                action = dist.rsample() #rsample includes the reparametrization trick
                action = torch.tanh(action)

            else:
                action = mean

            log_prob = dist.log_prob(action)

            actions.append(action) 
            log_probs.append(log_prob)

        #print("actions", actions)
        #print("log_probs", log_probs)

        if state.shape[0] == self.state_dim: # working with a single state
            
            actions = torch.tensor(actions).reshape((self.action_dim, ))
            log_probs = torch.tensor(log_probs).reshape((self.action_dim, ))

        else:  # second dimension is the state shape

            N = state.shape[0]
            actions = torch.tensor(actions).reshape((N, self.action_dim))
            log_probs = torch.tensor(log_probs).reshape((N, self.action_dim))


        #print("-------after conversion to tensors")
        #print("actions", actions)
        #print("log_probs", log_probs)

        assert (actions.shape == (self.action_dim, ) and \
            log_probs.shape == (self.action_dim, ), 'Incorrect shape for action or log_prob.' ) or \
                ( actions.shape[1] == self.action_dim and log_probs.shape[1] == self.action_dim )
             
        return actions, log_probs


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        #pass
        #We set the output to 1, but are not sure if the expected value returns a vector
        #self.NN_critic_lr = self.critic_lr
        self.NN_critic = NeuralNetwork(input_dim = self.state_dim, output_dim=1, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, activation="relu")
        self.NN_critic.to(self.device)
        self.optimizer = optim.Adam(self.NN_critic.parameters(),lr = self.critic_lr)
        #self.temperature = TrainableParameter(init_param=0.005, lr_param=0.1, train_param=True)
        #self.temperature = TrainableParameter(init_param=0.005, lr_param=0.1, train_param=True)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        #pass
        self.hidden_layers = 2
        self.hidden_size = 256
        self.lr = 3E-3

        self.actor = Actor(hidden_size = self.hidden_size,
                           hidden_layers = self.hidden_layers,
                           actor_lr = self.lr)
        self.critic_Q2 = Critic(state_dim = self.state_dim+self.action_dim,
                                hidden_size = self.hidden_size, 
                                hidden_layers = self.hidden_layers,
                                critic_lr = self.lr)
        
        self.critic_Q1 = Critic(state_dim=self.state_dim+self.action_dim,
                                hidden_size=self.hidden_size,
                                hidden_layers=self.hidden_layers,
                                critic_lr=self.lr)
        #self.critic = Critic(self.hidden_size, self.hidden_layers, self.lr)
        #self.trainable_params = TrainableParameter(init_param: float, self.lr, train_param: bool)
        #Name parameters from the paper
        #self.log_prob = []
        self.Tau = 0.005
        self.gamma = 0.99
        self.temperature = TrainableParameter(init_param=0.05, lr_param=0.01, train_param=True)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        #action = np.random.uniform(-1, 1, (1,))
        #Convert the state to a torch tensor, which is the required input for the actor
        s = torch.tensor(s)
        #Import action from the actor and discard the log probability here, possibly used elsewhere
        action, _ = self.actor.get_action_and_log_prob(s, False)
        # only get one action -> we have to sample in get_action_and_log_prob
        #Convert the returned tensor action to an nd.array
        action = action.clone().detach().numpy()
        #Need log probability for something -------> ?

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    # loss: 200 x 1
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self): 
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        #print("s_batch", s_batch[0:5, :])

        print("#############################")
        print("train_agent")
        print("#############################")

        #Get the temperature - We still need to figure out which network uses this
        alpha = self.temperature.get_param()
        print("alpha before optimization", alpha)
        #alpha = torch.tensor(0.5)
        reward =  1/alpha * r_batch # smth to investigate
        print("modified reward", reward[0:5, :])
        #reward = r_batch + alpha * entropy <--

        #Store the basic Psi network - which I guess we still need
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

        #print("Q1 before gradient", base_net1.state_dict()['putput.weight'][0,:5])

        #Optimize the critic networks
        #Run a gradient update step for critic V
        # TODO: Implement Critic(s) update here.

        print(" ------- Training Critic networks -------- ")
       
        with torch.no_grad():

            ###### OLD VERSION #######

            #results_list = [self.actor.get_action_and_log_prob(state, False) for state in s_prime_batch] 
            
            #next_sampled_action, next_sampled_log_prob = zip(*results_list)

            #next_sampled_action = torch.tensor(next_sampled_action).flatten().reshape(self.batch_size, 1)
            #next_sampled_log_prob = torch.tensor(next_sampled_log_prob).flatten().reshape(self.batch_size, 1)

            ###### NEW VERSION #########

            next_sampled_action, next_sampled_log_prob = self.actor.get_action_and_log_prob(s_prime_batch, False)

            print("next_sampled_action",next_sampled_action[0:5,:])
            print("next_sampled_log_prob", next_sampled_log_prob[0:5,:])

            input = torch.cat((s_prime_batch, next_sampled_action), dim = 1).to(self.device)
            #print("input looks like", input[0:5,])
            qf1_next = self.critic_Q1.NN_critic(input)   
            qf2_next = self.critic_Q2.NN_critic(input)

            #print("Total number of zero outputs", (qf1_next == 0).sum(), "out of", qf1_next.shape)
            min_qf_next = torch.min(qf1_next,qf2_next) - next_sampled_log_prob

            #print("min_qf_next",min_qf_next[0:5,:])
            next_q_value = reward + self.gamma * min_qf_next # 200 x 1

        #print("next_q_value", next_q_value[0:5,:])

        #Get the current values and optimize with respect to the next ones
        input_Q = torch.cat((s_batch, a_batch), dim = 1).to(self.device)
    
        qf1 = self.critic_Q1.NN_critic(input_Q) # 200 x 1
        qf2 = self.critic_Q2.NN_critic(input_Q) # 200 x 1

        #print("Total number of zero outputs", (qf1 == 0).sum(), "out of", qf1.shape)

        q1_loss = nn.functional.mse_loss(qf1, next_q_value)  
        q2_loss = nn.functional.mse_loss(qf2,next_q_value)

        print("q1 loss", q1_loss)

        self.run_gradient_update_step(self.critic_Q1, q1_loss)
        self.run_gradient_update_step(self.critic_Q2, q2_loss)

        # print some gradients

        #print("grad of NN_critic Q1 putput weight", self.critic_Q1.NN_critic.putput.weight.grad)
        #print("grad of NN_critic Q1 input weight", self.critic_Q1.NN_critic.input.weight.grad)
        #print("grad of NN_critic Q2", )

        #Sample current action and its log_prob

        
        print("-------- Training Policy Network ---------- ")

        with torch.no_grad():

            ###### OLD VERSION #######

            #results_list2 = [self.actor.get_action_and_log_prob(state, False) for state in s_batch]

            #sampled_action, sampled_log_prob = zip(*results_list2) #self.actor.get_action_and_log_prob(state=s_batch, deterministic=False)
        
            #sampled_action = torch.tensor(sampled_action).flatten().reshape(self.batch_size, 1)
            #sampled_log_prob = torch.tensor(sampled_log_prob).flatten().reshape(self.batch_size, 1)

            ####### NEW VERSION ######

            sampled_action, sampled_log_prob = self.actor.get_action_and_log_prob(s_batch, False)


        input_policy = torch.cat((s_batch, sampled_action), dim = 1).to(self.device)
        Q1_pi = self.critic_Q1.NN_critic(input_policy) #s_batch,sampled_action)
        Q2_pi = self.critic_Q2.NN_critic(input_policy) #s_batch,sampled_action)
        min_q_pi = torch.min(Q1_pi, Q2_pi)

        print("sampled actions are", sampled_action[0:5,:] )
        print("sampled log probs are", sampled_log_prob[0:5,:] )
        
        #Policy loss


        # TODO: Implement Policy update here
        policy_loss = ((sampled_log_prob) - min_q_pi) # self.alpha * removed
        print("policy loss", policy_loss[0:5,])

        #Gradient update for policy
        self.run_gradient_update_step(self.actor, policy_loss)

        print("grad of policy network", self.actor.NN_actor.input.weight.grad)

        # print some gradients

        # Temperature (alpha) loss
        print("------ Training temperature -------")

        H = -1.
        alpha_loss = - alpha * sampled_log_prob - alpha * H

        self.temperature.optimizer.zero_grad()
        alpha_loss.mean().backward()
        self.temperature.optimizer.step()
        
        #print("targetnet --> Q1 after gradient, before soft update", self.critic_Q1.NN_critic.state_dict()['putput.weight'][0,:5])
        #Critic target update step
        #print("basenet1 -->", base_net1.state_dict()['putput.weight'][0,:5])

        self.critic_target_update(base_net1, self.critic_Q1.NN_critic, self.Tau,True)
        self.critic_target_update(base_net2, self.critic_Q2.NN_critic, self.Tau,True)

        #print("Q1 after update", self.critic_Q1.NN_critic.state_dict()['putput.weight'][0,:5])

        #alpha = self.temperature.get_param()
        #print("alpha after optimization", alpha)
        


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 10
    TEST_EPISODES = 5

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        print("Running episode: ", EP)
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
