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
        s = self.activation(s)
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
        self.NN_actor = NeuralNetwork(input_dim=self.state_dim, output_dim=2*self.action_dim, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, activation="relu")
        self.NN_actor = optim.Adam(self.NN_actor.parameters(),lr = self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        mean, std = self.NN_actor(state)  #Returns an action and a probability from the neural network
        #How are probability and the log standard deviation related?
        if deterministic == False:  #We aren't sure about the placement of the clamping, as it makes a difference for the probability, what its std is
            log_std = self.clamp_log_std(np.log(std))   #The log of the standard deviation must be clamped not the standard deviation
        std = np.exp(log_std)
        #Sample an action from the Gaussian
        action = np.random.normal(mean,std)
        prob = norm(mean,std).pdf(action)   #Have to add scipy.stats.norm to requirements somehow
        log_prob = np.log(prob)
        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


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
        self.optimizer = optim.Adam(self.NN_critic.parameters(),lr = self.critic_lr)
        self.temperature = TrainableParameter(init_param=0.005, lr_param=0.1, train_param=True)


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
        self.lr = 3E-4
        self.actor = Actor(self.hidden_size, self.hidden_layers, self.lr)
        #We define separate critics
        self.critic_V = Critic(state_dim=self.state_dim,hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, critic_lr=self.lr)
        self.critic_Q = Critic(state_dim=self.state_dim+self.action_dim,hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, critic_lr=self.lr)
        self.Tau = 0.005

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
        action, _ = self.actor.get_action_and_log_prob(s, not(train))
        # only get one action -> we have to sample in get_action_and_log_prob
        #Convert the returned tensor action to an nd.array
        action = action.numpy()
        #Need log probability for something -------> ?

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
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

    def train_agent(self):  #------------> ? Christoph is a bit confused, but we need to implement phi, psi and theta gradient updates
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

        #Get the temperature - We still need to figure out which network uses this
        alpha = self.critic_Q.temperature.get_param()
        reward = 1/alpha * r_batch

        #Store the basic Psi network
        base_net = self.critic_V.NN_critic()

        #Run a gradient update step for critic V
        # TODO: Implement Critic(s) update here.
        loss_critic_V = 0
        loss_critic_Q = 0
        self.run_gradient_update_step(self.critic_V,loss_critic_V)
        self.run_gradient_update_step(self.critic_Q,loss_critic_Q)
        self.critic_target_update(base_net,self.critic_V.NN_critic,self.Tau,True)
        
        #Since we're performing an update step, we must include this new sampled batch in the neural network
        #reward_sampled = -(s_batch^2 + 0.1*s_prime_batch^2 + 0.001*a_batch^2)
        #self.actor.NN_actor.train() #Train the actor
        #self.critic.NN_critic_V.train() #Train the first critic
        #self.critic.NN_critic_Q.train() #Train the second critic

        # TODO: Implement Policy update here
        #policy = np.transpose(np.array([np.cos(s_batch),np.sin(s_batch),s_prime_batch]))
        loss_actor = 0
        self.run_gradient_update_step(self.actor, loss_actor)

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
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
