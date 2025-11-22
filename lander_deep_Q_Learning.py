import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

import numpy as np
from collections import deque, namedtuple

# For visualization
from gymnasium.wrappers import RecordVideo as video_recorder
from IPython.display import HTML
from IPython import display 
import glob

env = gym.make('LunarLander-v3')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def moving_average(data, window_size):
    """Calculate moving average with given window size."""
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def evaluate_agent(agent, env, n_episodes=100):
    """Evaluate agent without exploration noise."""
    eval_scores = []
    successes = 0
    
    for _ in range(n_episodes):
        state, info = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state, eps=0.0)  # No exploration
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
        
        eval_scores.append(score)
        if score >= 200:
            successes += 1
    
    return np.mean(eval_scores), successes / n_episodes * 100, eval_scores


def dqn(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        eval_frequency=100, eval_episodes=20):
    """Deep Q-Learning with comprehensive metrics tracking.
    
    Params
    ======
        agent: The DQN agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor for decreasing epsilon
        eval_frequency (int): how often to run evaluation
        eval_episodes (int): number of episodes per evaluation
    
    Returns
    =======
        dict: Training metrics including scores, evaluations, success rates
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # Metrics for comprehensive plotting
    eval_episodes_list = []
    eval_mean_rewards = []
    eval_success_rates = []
    
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        # Periodic evaluation
        if i_episode % eval_frequency == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
            # Run evaluation
            eval_mean, eval_success, _ = evaluate_agent(agent, env, n_episodes=eval_episodes)
            eval_episodes_list.append(i_episode)
            eval_mean_rewards.append(eval_mean)
            eval_success_rates.append(eval_success)
            
            print(f'  Evaluation: Mean Reward = {eval_mean:.2f}, Success Rate = {eval_success:.1f}%')
        
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    # Final evaluation
    print('\nRunning final evaluation (100 episodes)...')
    final_mean, final_success, final_scores = evaluate_agent(agent, env, n_episodes=100)
    print(f'Final Performance: Mean Reward = {final_mean:.2f}, Success Rate = {final_success:.1f}%')
    
    return {
        'scores': scores,
        'eval_episodes': eval_episodes_list,
        'eval_mean_rewards': eval_mean_rewards,
        'eval_success_rates': eval_success_rates,
        'final_scores': final_scores,
        'final_mean': final_mean,
        'final_success': final_success
    }


def create_comprehensive_plot(metrics, algorithm_name='DQN', ma_window=500, save_path='dqn_comprehensive_plot.png'):
    """
    Create a comprehensive 6-panel plot similar to the tabular RL comparison plots.
    
    Params
    ======
        metrics (dict): Training metrics from dqn() function
        algorithm_name (str): Name of the algorithm for labels
        ma_window (int): Window size for moving average
        save_path (str): Path to save the plot
    """
    scores = metrics['scores']
    eval_episodes = metrics['eval_episodes']
    eval_mean_rewards = metrics['eval_mean_rewards']
    eval_success_rates = metrics['eval_success_rates']
    final_scores = metrics['final_scores']
    final_mean = metrics['final_mean']
    final_success = metrics['final_success']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{algorithm_name} Training Analysis - LunarLander-v3', fontsize=14, fontweight='bold')
    
    # Color for single algorithm
    color = '#1f77b4'  # Blue
    
    # Success threshold line
    success_threshold = 200
    
    # ===== Plot 1: Training Progress (Moving Average) =====
    ax1 = axes[0, 0]
    ma_scores = moving_average(scores, ma_window)
    episodes_ma = np.arange(ma_window, len(scores) + 1)
    
    ax1.plot(episodes_ma, ma_scores, color=color, label=algorithm_name, linewidth=1.5)
    ax1.axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(f'Reward (MA-{ma_window})')
    ax1.set_title('Training Progress')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Evaluation Performance =====
    ax2 = axes[0, 1]
    ax2.plot(eval_episodes, eval_mean_rewards, color=color, marker='o', 
             markersize=4, label=algorithm_name, linewidth=1.5)
    ax2.axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Evaluation Reward')
    ax2.set_title('Evaluation Performance')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Success Rate Over Time =====
    ax3 = axes[0, 2]
    ax3.plot(eval_episodes, eval_success_rates, color=color, marker='o',
             markersize=4, label=algorithm_name, linewidth=1.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Landing Success Rate')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # ===== Plot 4: Reward Distribution (Final Episodes) =====
    ax4 = axes[1, 0]
    n_final = min(10000, len(scores))
    final_training_scores = scores[-n_final:]
    
    ax4.hist(final_training_scores, bins=50, color=color, alpha=0.7, 
             label=algorithm_name, edgecolor='black', linewidth=0.5)
    ax4.axvline(x=success_threshold, color='green', linestyle='--', 
                label='Success', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Reward Distribution (Final {n_final//1000}K Episodes)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Learning Curves (Final Episodes) =====
    ax5 = axes[1, 1]
    n_final_curve = min(100000, len(scores))
    final_curve_scores = scores[-n_final_curve:]
    ma_window_final = min(1000, len(final_curve_scores) // 10)
    
    if ma_window_final > 0:
        ma_final = moving_average(final_curve_scores, ma_window_final)
        episodes_final = np.arange(len(scores) - n_final_curve + ma_window_final, len(scores) + 1)
        ax5.plot(episodes_final, ma_final, color=color, label=algorithm_name, linewidth=1.5)
    
    ax5.axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel(f'Reward (MA-{ma_window_final})')
    ax5.set_title(f'Learning Curves (Final {n_final_curve//1000}K Episodes)')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    # ===== Plot 6: Final Performance Comparison =====
    ax6 = axes[1, 2]
    
    # Create bar chart
    x = np.array([0])
    width = 0.35
    
    # Mean reward bar
    bars1 = ax6.bar(x - width/2, [final_mean], width, label='Mean Reward', color=color)
    
    # Create second y-axis for success rate
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x + width/2, [final_success], width, label='Success Rate (%)', 
                         color=color, alpha=0.5)
    
    ax6.axhline(y=success_threshold, color='green', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Mean Reward (Final 100 eps)')
    ax6_twin.set_ylabel('Success Rate (%)')
    ax6.set_title('Final Performance')
    ax6.set_xticks(x)
    ax6.set_xticklabels([algorithm_name])
    ax6.set_ylim(-50, 250)
    ax6_twin.set_ylim(0, 100)
    
    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add value labels on bars
    ax6.bar_label(bars1, fmt='%.1f', padding=3)
    ax6_twin.bar_label(bars2, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nPlot saved to: {save_path}')


def create_multi_algorithm_comparison(all_metrics, save_path='algorithm_comparison.png'):
    """
    Create a comparison plot for multiple algorithms (like your tabular RL comparisons).
    
    Params
    ======
        all_metrics (dict): Dictionary with algorithm names as keys and metrics dicts as values
                           e.g., {'DQN': metrics_dqn, 'Double DQN': metrics_ddqn}
        save_path (str): Path to save the plot
    """
    # Colors for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Algorithm Comparison - LunarLander-v3', fontsize=14, fontweight='bold')
    
    success_threshold = 200
    ma_window = 500
    
    for idx, (algo_name, metrics) in enumerate(all_metrics.items()):
        color = colors[idx % len(colors)]
        scores = metrics['scores']
        eval_episodes = metrics['eval_episodes']
        eval_mean_rewards = metrics['eval_mean_rewards']
        eval_success_rates = metrics['eval_success_rates']
        
        # Plot 1: Training Progress
        ma_scores = moving_average(scores, ma_window)
        episodes_ma = np.arange(ma_window, len(scores) + 1)
        axes[0, 0].plot(episodes_ma, ma_scores, color=color, label=algo_name, linewidth=1.5)
        
        # Plot 2: Evaluation Performance
        axes[0, 1].plot(eval_episodes, eval_mean_rewards, color=color, marker='o',
                       markersize=4, label=algo_name, linewidth=1.5)
        
        # Plot 3: Success Rate
        axes[0, 2].plot(eval_episodes, eval_success_rates, color=color, marker='o',
                       markersize=4, label=algo_name, linewidth=1.5)
        
        # Plot 4: Reward Distribution
        n_final = min(10000, len(scores))
        final_scores = scores[-n_final:]
        axes[1, 0].hist(final_scores, bins=50, color=color, alpha=0.5,
                       label=algo_name, edgecolor='black', linewidth=0.5)
        
        # Plot 5: Final Learning Curves
        n_final_curve = min(100000, len(scores))
        final_curve_scores = scores[-n_final_curve:]
        ma_window_final = min(1000, len(final_curve_scores) // 10)
        if ma_window_final > 0:
            ma_final = moving_average(final_curve_scores, ma_window_final)
            episodes_final = np.arange(len(scores) - n_final_curve + ma_window_final, len(scores) + 1)
            axes[1, 1].plot(episodes_final, ma_final, color=color, label=algo_name, linewidth=1.5)
    
    # Add success threshold lines
    axes[0, 0].axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    axes[0, 1].axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    axes[1, 0].axvline(x=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    axes[1, 1].axhline(y=success_threshold, color='green', linestyle='--', label='Success', alpha=0.7)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel(f'Reward (MA-{ma_window})')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Mean Evaluation Reward')
    axes[0, 1].set_title('Evaluation Performance')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Success Rate (%)')
    axes[0, 2].set_title('Landing Success Rate')
    axes[0, 2].legend(loc='lower right')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 105)
    
    axes[1, 0].set_xlabel('Episode Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution (Final 10K Episodes)')
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward (MA-1000)')
    axes[1, 1].set_title('Learning Curves (Final 100K Episodes)')
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final Performance Bar Chart
    ax6 = axes[1, 2]
    algo_names = list(all_metrics.keys())
    x = np.arange(len(algo_names))
    width = 0.35
    
    final_means = [all_metrics[name]['final_mean'] for name in algo_names]
    final_successes = [all_metrics[name]['final_success'] for name in algo_names]
    
    bars1 = ax6.bar(x - width/2, final_means, width, label='Mean Reward', 
                    color=[colors[i % len(colors)] for i in range(len(algo_names))])
    
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x + width/2, final_successes, width, label='Success Rate (%)',
                         color=[colors[i % len(colors)] for i in range(len(algo_names))], alpha=0.5)
    
    ax6.axhline(y=success_threshold, color='green', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Mean Reward (Final 100 eps)')
    ax6_twin.set_ylabel('Success Rate (%)')
    ax6.set_title('Final Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(algo_names)
    ax6.set_ylim(-50, 250)
    ax6_twin.set_ylim(0, 100)
    
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nPlot saved to: {save_path}')


def train_and_plot():
    """Train DQN agent and create comprehensive plot."""
    agent = Agent(state_size=8, action_size=4, seed=0)
    
    # Train with metrics collection
    metrics = dqn(
        agent,
        n_episodes=2000,       # Adjust as needed
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        eval_frequency=100,    # Evaluate every 100 episodes
        eval_episodes=20       # 20 episodes per evaluation
    )
    
    # Create comprehensive plot
    create_comprehensive_plot(
        metrics, 
        algorithm_name='DQN',
        ma_window=100,  # Smaller window for fewer episodes
        save_path='dqn_comprehensive_plot.png'
    )
    
    return agent, metrics


def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    vid = video_recorder(env, video_folder="video")

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state, info = env.reset()
    done = False
    vid.start_recording(env_name)
    episodes = 1
    while episodes < 5:
        frame = env.render()
        vid._capture_frame()
        
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:   
            episodes += 1
            state, info = env.reset()
        else:
            state = next_state 
    env.close()
    vid.stop_recording()


if __name__ == "__main__":
    agent, metrics = train_and_plot()
    
    # show_video_of_model(agent, 'LunarLander-v3')
    # show_video('LunarLander-v3')