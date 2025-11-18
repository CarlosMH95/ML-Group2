import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import pickle
"""
THREE ALGORITHMS
====================================================

Comparing three fundamental RL algorithms:
1. Q-Learning (Off-policy TD)
2. SARSA (On-policy TD)
3. Monte Carlo (Episode-based)
"""
N_BINS = 12  

LEARNING_RATE = 0.1 
DISCOUNT = 0.99
EPISODES = 500000        

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99999

USE_REWARD_SHAPING = True
SHAPING_WEIGHT = 0.15   

EVAL_FREQUENCY = 5000
EVAL_EPISODES = 20

# ============================================================================
# IMPROVED STATE DISCRETIZATION
# ============================================================================

# More bins = better precision
# Focus ranges on critical regions

# X-position: More bins near center (where pad is)
x_pos_bins = np.array([-1.5, -0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8, 1.5])

# Y-position: More bins near ground (where landing happens)
y_pos_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0])

# Velocities: Standard bins
x_vel_bins = np.linspace(-2.0, 2.0, N_BINS - 1)
y_vel_bins = np.linspace(-2.0, 1.0, N_BINS - 1)

# Angle: More bins near upright
angle_bins = np.array([-0.8, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8])

# Angular velocity: Standard
ang_vel_bins = np.linspace(-2.0, 2.0, N_BINS - 1)

N_ACTIONS = 4


def discretize_state(state):
    """Better discretization with non-uniform bins."""
    x_pos = np.clip(state[0], -1.5, 1.5)
    y_pos = np.clip(state[1], 0.0, 2.0)
    x_vel = np.clip(state[2], -2.0, 2.0)
    y_vel = np.clip(state[3], -2.0, 1.0)
    angle = np.clip(state[4], -0.8, 0.8)
    ang_vel = np.clip(state[5], -2.0, 2.0)
    
    s0 = np.digitize(x_pos, x_pos_bins)
    s1 = np.digitize(y_pos, y_pos_bins)
    s2 = np.digitize(x_vel, x_vel_bins)
    s3 = np.digitize(y_vel, y_vel_bins)
    s4 = np.digitize(angle, angle_bins)
    s5 = np.digitize(ang_vel, ang_vel_bins)
    
    #return (s0, s1, s2, s3, s4, s5)
    return (s1, s3, s4, s5)  # Use only critical dimensions


# ============================================================================
# REWARD SHAPING
# ============================================================================

def simple_reward_shaping(state, reward, done):
    """
    SIMPLE shaping: just guide toward safe landing configuration.
    
    1. Being upright
    2. Low velocity when near ground
    3. Centered position when low
    """
    if not USE_REWARD_SHAPING:
        return reward
    
    shaped_reward = reward
    
    y_pos = state[1]
    #x_vel = state[2]
    y_vel = state[3]
    angle = state[4] 
    # Simple shaping: reward good landing configuration
    if y_pos < 0.5:  # Only when low (near landing)
        # Reward being upright
        if abs(angle) < 0.1:
            shaped_reward += 0.1 * SHAPING_WEIGHT
        
        # Reward low velocity
        if abs(y_vel) < 0.3:
            shaped_reward += 0.1 * SHAPING_WEIGHT
        
        # Reward being centered
        if abs(state[0]) < 0.2:
            shaped_reward += 0.1 * SHAPING_WEIGHT
    
    return shaped_reward
# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Base class for all RL agents."""
    
    def __init__(self, n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay):
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        
        # Standard initialization to zero
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.state_visits = defaultdict(int)
        self.total_updates = 0
        self.agent_name = "BaseAgent"

    def choose_action(self, state_tuple, evaluation=False):
        """Standard epsilon-greedy."""
        if evaluation:
            return np.argmax(self.q_table[state_tuple])
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_table[state_tuple])

    def decay_epsilon(self):
        """Standard exponential decay."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent(BaseAgent):
    """
    Q-Learning: Off-policy TD control.
    Updates using max Q-value of next state.
    """
    
    def __init__(self, n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay):
        super().__init__(n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay)
        self.agent_name = "Q-Learning"

    def update(self, state, action, reward, next_state, done):
        """Standard Q-Learning update (off-policy)."""
        current_q = self.q_table[state][action]
        
        if done:
            max_future_q = 0.0
        else:
            # Q-Learning: use MAX Q-value of next state (off-policy)
            max_future_q = np.max(self.q_table[next_state])
        
        # TD(0) update
        td_target = reward + self.discount * max_future_q
        new_q = current_q + self.lr * (td_target - current_q)
        
        self.q_table[state][action] = new_q
        self.state_visits[state] += 1
        self.total_updates += 1


# ============================================================================
# SARSA AGENT
# ============================================================================

class SARSAAgent(BaseAgent):
    """
    SARSA: On-policy TD control.
    Updates using Q-value of actual next action taken.
    """
    
    def __init__(self, n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay):
        super().__init__(n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay)
        self.agent_name = "SARSA"

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update (on-policy)."""
        current_q = self.q_table[state][action]
        
        if done:
            future_q = 0.0
        else:
            # SARSA: use Q-value of ACTUAL next action (on-policy)
            future_q = self.q_table[next_state][next_action]
        
        # TD(0) update
        td_target = reward + self.discount * future_q
        new_q = current_q + self.lr * (td_target - current_q)
        
        self.q_table[state][action] = new_q
        self.state_visits[state] += 1
        self.total_updates += 1


# ============================================================================
# MONTE CARLO AGENT
# ============================================================================

class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo: Episode-based learning.
    Updates Q-values after complete episodes using actual returns.
    
    Using first-visit MC with averaging.
    """
    
    def __init__(self, n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay):
        super().__init__(n_actions, learning_rate, discount, epsilon_start, epsilon_end, epsilon_decay)
        self.agent_name = "Monte Carlo"
        
        # Track returns for each (state, action) pair
        self.returns = defaultdict(list)
        
        # Current episode trajectory
        self.episode_trajectory = []
    
    def start_episode(self):
        """Reset trajectory at the start of each episode."""
        self.episode_trajectory = []
    
    def store_transition(self, state, action, reward):
        """Store transition during episode."""
        self.episode_trajectory.append((state, action, reward))
    
    def update_from_episode(self):
        """
        Update Q-values after episode completes.
        Uses first-visit Monte Carlo with incremental averaging.
        """
        if len(self.episode_trajectory) == 0:
            return
        
        # Calculate returns for each step (working backwards)
        G = 0  # Return
        visited_state_actions = set()
        
        # Process trajectory in reverse
        for t in reversed(range(len(self.episode_trajectory))):
            state, action, reward = self.episode_trajectory[t]
            
            # Update return
            G = reward + self.discount * G
            
            # First-visit MC: only update if this (s,a) pair hasn't been seen yet in this episode
            state_action = (state, action)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                
                # Store return
                self.returns[state_action].append(G)
                
                # Update Q-value using incremental average
                # Q(s,a) = average of all returns following (s,a)
                current_q = self.q_table[state][action]
                # Using incremental update for efficiency
                n = len(self.returns[state_action])
                new_q = current_q + (G - current_q) / n
                
                self.q_table[state][action] = new_q
                self.state_visits[state] += 1
                self.total_updates += 1


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_agent(agent, n_episodes=20, render=False):
    """Standard evaluation."""
    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        env = gym.make('LunarLander-v3')
    
    episode_rewards = []
    successes = 0
    
    for _ in range(n_episodes):
        state_full, _ = env.reset()
        state = discretize_state(state_full)
        
        total_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent.choose_action(state, evaluation=True)
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        if total_reward >= 200:
            successes += 1
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'success_rate': successes / n_episodes * 100
    }


# ============================================================================
# UNIFIED TRAINING FUNCTION
# ============================================================================

def train_agent(agent, agent_type, verbose=True):
    """
    Universal training function for all three agent types.
    
    Args:
        agent: Agent instance (QLearning, SARSA, or MonteCarlo)
        agent_type: String identifier ('q-learning', 'sarsa', or 'monte-carlo')
        verbose: Whether to print progress
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {agent.agent_name.upper()}")
    print(f"{'='*70}\n")
    
    env = gym.make('LunarLander-v3')
    
    training_rewards = []
    evaluation_stats = []
    start_time = time.time()
    
    for episode in range(EPISODES):
        state_full, _ = env.reset()
        state = discretize_state(state_full)
        
        total_reward = 0
        done = False
        truncated = False
        
        # Monte Carlo: start new episode
        if agent_type == 'monte-carlo':
            agent.start_episode()
        
        # SARSA: choose first action before loop
        if agent_type == 'sarsa':
            action = agent.choose_action(state)
        
        while not done and not truncated:
            # Q-Learning and Monte Carlo: choose action inside loop
            if agent_type in ['q-learning', 'monte-carlo']:
                action = agent.choose_action(state)
            
            # Take action
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            
            # Apply reward shaping
            shaped_reward = simple_reward_shaping(next_state_full, reward, done or truncated)
            
            # Update based on agent type
            if agent_type == 'q-learning':
                agent.update(state, action, shaped_reward, next_state, done or truncated)
                
            elif agent_type == 'sarsa':
                next_action = agent.choose_action(next_state)
                agent.update(state, action, shaped_reward, next_state, next_action, done or truncated)
                action = next_action  # Use this action in next iteration
                
            elif agent_type == 'monte-carlo':
                # Store transition for end-of-episode update
                agent.store_transition(state, action, shaped_reward)
            
            state = next_state
            state_full = next_state_full
            total_reward += reward  # Use original reward for tracking

        # Monte Carlo: update Q-values after episode
        if agent_type == 'monte-carlo':
            agent.update_from_episode()

        agent.decay_epsilon()
        training_rewards.append(total_reward)
        
        # Evaluation
        if (episode + 1) % EVAL_FREQUENCY == 0:
            eval_stats = evaluate_agent(agent, n_episodes=EVAL_EPISODES)
            evaluation_stats.append({'episode': episode + 1, **eval_stats})
            
            elapsed = time.time() - start_time
            states_visited = len(agent.state_visits)
            
            if verbose:
                print(f"Ep {episode + 1:6d}/{EPISODES} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Eval: {eval_stats['mean_reward']:7.1f} | "
                      f"Success: {eval_stats['success_rate']:5.1f}% | "
                      f"States: {states_visited:7d} | "
                      f"Time: {elapsed/60:5.1f}m")
        
        elif verbose and (episode + 1) % 10000 == 0:
            recent_avg = np.mean(training_rewards[-5000:])
            print(f"Ep {episode + 1:6d} | Recent avg: {recent_avg:7.1f} | ε: {agent.epsilon:.3f}")
    
    env.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: {agent.agent_name}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"States visited: {len(agent.state_visits):,}")
    print(f"Total updates: {agent.total_updates:,}")
    print(f"{'='*70}\n")
    
    return training_rewards, evaluation_stats


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_comparison(results_dict):
    """
    Create comprehensive comparison plots for all three agents.
    
    Args:
        results_dict: Dictionary with keys 'q-learning', 'sarsa', 'monte-carlo'
                     Each value is a tuple of (rewards, eval_stats)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {
        'q-learning': '#1f77b4',
        'sarsa': '#ff7f0e',
        #'monte-carlo': '#2ca02c'
    }
    
    labels = {
        'q-learning': 'Q-Learning',
        'sarsa': 'SARSA',
        #'monte-carlo': 'Monte Carlo'
    }
    
    # 1. Training rewards (moving average)
    window = 500
    ax = axes[0, 0]
    for agent_type, (rewards, _) in results_dict.items():
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(moving_avg, label=labels[agent_type], 
                color=colors[agent_type], alpha=0.8, linewidth=2)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Success')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Reward (MA-{window})', fontsize=11)
    ax.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Evaluation mean rewards
    ax = axes[0, 1]
    for agent_type, (_, eval_stats) in results_dict.items():
        eval_episodes = [stat['episode'] for stat in eval_stats]
        eval_means = [stat['mean_reward'] for stat in eval_stats]
        ax.plot(eval_episodes, eval_means, marker='o', 
                label=labels[agent_type], color=colors[agent_type], 
                alpha=0.8, linewidth=2, markersize=4)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Mean Evaluation Reward', fontsize=11)
    ax.set_title('Evaluation Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Success rate over time
    ax = axes[0, 2]
    for agent_type, (_, eval_stats) in results_dict.items():
        eval_episodes = [stat['episode'] for stat in eval_stats]
        success_rates = [stat['success_rate'] for stat in eval_stats]
        ax.plot(eval_episodes, success_rates, marker='s', 
                label=labels[agent_type], color=colors[agent_type],
                alpha=0.8, linewidth=2, markersize=4)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Landing Success Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Reward distribution (final 10K episodes)
    ax = axes[1, 0]
    for agent_type, (rewards, _) in results_dict.items():
        ax.hist(rewards[-10000:], bins=40, alpha=0.5, 
                label=labels[agent_type], color=colors[agent_type])
    ax.axvline(x=200, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Episode Reward', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Reward Distribution (Final 10K Episodes)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Learning curves (last 100K episodes)
    ax = axes[1, 1]
    start_ep = max(0, len(list(results_dict.values())[0][0]) - 100000)
    window = 1000
    for agent_type, (rewards, _) in results_dict.items():
        recent_rewards = rewards[start_ep:]
        moving_avg = np.convolve(recent_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(start_ep, start_ep + len(moving_avg)), moving_avg,
                label=labels[agent_type], color=colors[agent_type],
                alpha=0.8, linewidth=2)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Reward (MA-{window})', fontsize=11)
    ax.set_title('Learning Curves (Final 100K Episodes)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 6. Final performance comparison (bar chart)
    ax = axes[1, 2]
    agent_names = [labels[at] for at in results_dict.keys()]
    final_means = [np.mean(rewards[-10000:]) for rewards, _ in results_dict.values()]
    final_success = [eval_stats[-1]['success_rate'] if eval_stats else 0 
                     for _, eval_stats in results_dict.values()]
    
    x = np.arange(len(agent_names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_means, width, label='Mean Reward',
                   color=[colors[at] for at in results_dict.keys()], alpha=0.7)
    bars2 = ax2.bar(x + width/2, final_success, width, label='Success Rate (%)',
                    color=[colors[at] for at in results_dict.keys()], alpha=0.4)
    
    ax.set_xlabel('Algorithm', fontsize=11)
    ax.set_ylabel('Mean Reward (Final 10K eps)', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'outputs/4D/three_agents_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: three_agents_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LUNAR LANDER ")
    print("="*70)
    print("\nAlgorithms:")
    print("  1. Q-Learning: Off-policy TD (uses max Q of next state)")
    print("  2. SARSA: On-policy TD (uses Q of actual next action)")
    # Dictionary to store results
    all_results = {}
    all_agents = {}
    
    # =======================================================================
    # TRAIN Q-LEARNING
    # =======================================================================
    print("\n[1/2] Training Q-Learning Agent...")
    q_agent = QLearningAgent(
        n_actions=N_ACTIONS,
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY
    )
    rewards_q, eval_q = train_agent(q_agent, 'q-learning')
    all_results['q-learning'] = (rewards_q, eval_q)
    all_agents['q-learning'] = q_agent
    
    # Save Q-Learning model
    q_agent.save_q_table('outputs/4D/q_learning_model.pkl')
    print("Q-Learning model saved!")
    
    # =======================================================================
    # TRAIN SARSA
    # =======================================================================
    print("\n[2/2] Training SARSA Agent...")
    sarsa_agent = SARSAAgent(
        n_actions=N_ACTIONS,
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY
    )
    rewards_sarsa, eval_sarsa = train_agent(sarsa_agent, 'sarsa')
    all_results['sarsa'] = (rewards_sarsa, eval_sarsa)
    all_agents['sarsa'] = sarsa_agent
    
    # Save SARSA model
    sarsa_agent.save_q_table('outputs/4D/sarsa_model.pkl')
    print("SARSA model saved!")
    
    # =======================================================================
    # TRAIN MONTE CARLO
    # =======================================================================
    # print("\n[3/3] Training Monte Carlo Agent...")
    # mc_agent = MonteCarloAgent(
    #     n_actions=N_ACTIONS,
    #     learning_rate=LEARNING_RATE,  # Not used in MC but kept for consistency
    #     discount=DISCOUNT,
    #     epsilon_start=EPSILON_START,
    #     epsilon_end=EPSILON_END,
    #     epsilon_decay=EPSILON_DECAY
    # )
    # rewards_mc, eval_mc = train_agent(mc_agent, 'monte-carlo')
    # all_results['monte-carlo'] = (rewards_mc, eval_mc)
    # all_agents['monte-carlo'] = mc_agent
    
    # # Save Monte Carlo model
    # mc_agent.save_q_table('outputs/4D/monte_carlo_model.pkl')
    # print("Monte Carlo model saved!")
    
    # =======================================================================
    # FINAL EVALUATION AND COMPARISON
    # =======================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION (100 episodes each)")
    print("="*70)
    
    for agent_type, agent in all_agents.items():
        final_eval = evaluate_agent(agent, n_episodes=1000)
        print(f"\n{agent.agent_name}:")
        print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Success Rate: {final_eval['success_rate']:.1f}%")
        print(f"  Best Episode: {final_eval['max_reward']:.2f}")
        print(f"  States Explored: {len(agent.state_visits):,}")
    
    # =======================================================================
    # CREATE COMPARISON PLOTS
    # =======================================================================
    print("\n" + "="*70)
    print("Generating comparison plots...")
    plot_comparison(all_results)
    
    # =======================================================================
    # SUMMARY STATISTICS
    # =======================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nFinal 10,000 Episodes Performance:")
    print(f"{'Algorithm':<15} {'Mean Reward':>12} {'Std Dev':>10} {'Success %':>10}")
    print("-" * 50)
    for agent_type in ['q-learning', 'sarsa']:
        rewards, _ = all_results[agent_type]
        final_rewards = rewards[-10000:]
        mean_r = np.mean(final_rewards)
        std_r = np.std(final_rewards)
        success = np.sum(np.array(final_rewards) >= 200) / len(final_rewards) * 100
        print(f"{agent_type} {mean_r:>12.2f} {std_r:>10.2f} {success:>9.1f}%")
    
    # =======================================================================
    # OPTIONAL: WATCH TRAINED AGENT
    # =======================================================================
    print("\n" + "="*70)
    response = input("Would you like to watch one of the trained agents? (q/s/n): ")
    
    if response.lower() == 'q':
        print("Rendering Q-Learning agent (5 episodes)...")
        evaluate_agent(q_agent, n_episodes=5, render=True)
    elif response.lower() == 's':
        print("Rendering SARSA agent (5 episodes)...")
        evaluate_agent(sarsa_agent, n_episodes=5, render=True)
    # elif response.lower() == 'm':
    #     print("Rendering Monte Carlo agent (5 episodes)...")
    #     evaluate_agent(mc_agent, n_episodes=5, render=True)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nFiles saved:")
