"""
LunarLander - Trained Agent Demonstration
==========================================
This script loads saved Q-tables and visualizes the trained agents in action.

Usage:
    python demo_trained_agents.py
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pickle
from collections import defaultdict
import time
import os


# ============================================================================
# STATE DISCRETIZATION (Must match training)
# ============================================================================

N_BINS = 12

# Bins (must match training exactly)
x_pos_bins = np.linspace(-1.5, 1.5, N_BINS - 1)
y_pos_bins = np.linspace(0.0, 1.5, N_BINS - 1)
x_vel_bins = np.linspace(-2.0, 2.0, N_BINS - 1)
y_vel_bins = np.linspace(-2.0, 1.0, N_BINS - 1)
angle_bins = np.linspace(-0.8, 0.8, N_BINS - 1)
ang_vel_bins = np.linspace(-2.0, 2.0, N_BINS - 1)

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
    return (s0, s1, s2, s3, s4, s5)


# ============================================================================
# AGENT CLASS FOR LOADING AND USING TRAINED MODELS
# ============================================================================

class TrainedAgent:
    """Simple agent wrapper for loading and using trained Q-tables."""
    
    def __init__(self, q_table_path, name="Agent"):
        """
        Load a trained Q-table from file.
        
        Args:
            q_table_path: Path to the saved Q-table pickle file
            name: Name of the agent (for display purposes)
        """
        self.name = name
        self.n_actions = N_ACTIONS
        
        # Load the Q-table
        print(f"Loading {name} from: {q_table_path}")
        with open(q_table_path, 'rb') as f:
            loaded_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
            self.q_table.update(loaded_table)
        
        print(f"  ✓ Loaded {len(self.q_table)} state-action pairs")
    
    def choose_action(self, state_tuple):
        """
        Choose the best action (greedy policy - no exploration).
        
        Args:
            state_tuple: Discretized state
            
        Returns:
            int: Action index
        """
        return np.argmax(self.q_table[state_tuple])
    
    def get_action_name(self, action):
        """Get human-readable action name."""
        action_names = {
            0: "Do Nothing",
            1: "Fire Left Engine",
            2: "Fire Main Engine", 
            3: "Fire Right Engine"
        }
        return action_names.get(action, "Unknown")


# ============================================================================
# VIDEO RECORDING FUNCTION
# ============================================================================

def record_agent_video(agent, n_episodes=5, video_folder="videos", name_prefix=None):
    """
    Record video of trained agent playing LunarLander.
    
    Args:
        agent: TrainedAgent instance with Q-table
        n_episodes: Number of episodes to record
        video_folder: Directory to save videos
        name_prefix: Optional prefix for video filename (defaults to agent name)
    
    Returns:
        dict: Summary statistics of recorded episodes
    """
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Set up video filename prefix
    if name_prefix is None:
        name_prefix = agent.name.lower().replace(" ", "_").replace("-", "_")
    
    print(f"\n{'='*70}")
    print(f"RECORDING VIDEO: {agent.name}")
    print(f"{'='*70}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Output folder: {video_folder}/")
    print(f"  Video prefix: {name_prefix}")
    print()
    
    # Create environment with rgb_array render mode for video recording
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    
    # Wrap with RecordVideo - records all episodes
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: True  # Record all episodes
    )
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(n_episodes):
        print(f"  Recording episode {episode + 1}/{n_episodes}...", end=" ")
        
        state_full, _ = env.reset()
        state = discretize_state(state_full)
        
        total_reward = 0
        step_count = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Choose action using greedy policy
            action = agent.choose_action(state)
            
            # Take action
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            
            state = next_state
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        
        # Status indicator
        if total_reward >= 200:
            status = "✓ Success"
        elif total_reward > 0:
            status = "○ Partial"
        else:
            status = "✗ Crash"
        
        print(f"{status} (Reward: {total_reward:.1f}, Steps: {step_count})")
    
    env.close()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VIDEO RECORDING COMPLETE")
    print(f"{'='*70}")
    print(f"  Videos saved to: {video_folder}/")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Best Episode: {np.max(episode_rewards):.2f}")
    print(f"  Success Rate: {sum(1 for r in episode_rewards if r >= 200) / n_episodes * 100:.1f}%")
    print(f"{'='*70}\n")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'success_rate': sum(1 for r in episode_rewards if r >= 200) / n_episodes * 100,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'video_folder': video_folder
    }


def record_comparison_video(agents, n_episodes=3, video_folder="videos"):
    """
    Record videos of multiple agents for comparison.
    
    Args:
        agents: List of TrainedAgent instances
        n_episodes: Number of episodes per agent
        video_folder: Directory to save videos
    
    Returns:
        dict: Results for each agent
    """
    print(f"\n{'='*70}")
    print(f"RECORDING COMPARISON VIDEOS")
    print(f"{'='*70}")
    print(f"  Agents: {', '.join(a.name for a in agents)}")
    print(f"  Episodes per agent: {n_episodes}")
    print()
    
    results = {}
    
    for agent in agents:
        results[agent.name] = record_agent_video(
            agent, 
            n_episodes=n_episodes, 
            video_folder=video_folder
        )
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    for name, stats in results.items():
        print(f"  {name}:")
        print(f"    Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"    Success Rate: {stats['success_rate']:.1f}%")
    print(f"{'='*70}\n")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
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
            action = agent.choose_action(state)
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

def demonstrate_agent(agent, n_episodes=3, delay=0.01):
    """
    Visualize trained agent playing LunarLander.
    
    Args:
        agent: TrainedAgent instance
        n_episodes: Number of episodes to show
        delay: Delay between frames (seconds) for slower playback
    """
    print(f"\n{'='*70}")
    print(f"DEMONSTRATING: {agent.name}")
    print(f"{'='*70}\n")
    
    # Create environment with rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        state_full, _ = env.reset()
        state = discretize_state(state_full)
        
        total_reward = 0
        step_count = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Choose action (greedy - no exploration)
            action = agent.choose_action(state)
            
            # Display action being taken
            if step_count % 10 == 0:  # Print every 10 steps to avoid spam
                print(f"  Step {step_count}: {agent.get_action_name(action)}")
            
            # Take action in environment
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            
            # Add small delay for better viewing
            if delay > 0:
                time.sleep(delay)
            
            state = next_state
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        
        # Episode summary
        if total_reward >= 200:
            status = "✓ SUCCESSFUL LANDING!"
        elif total_reward > 0:
            status = "○ Partial success"
        else:
            status = "✗ Crashed"
        
        print(f"\n  {status}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")
        
        # Pause between episodes
        if episode < n_episodes - 1:
            time.sleep(1.5)
    
    env.close()
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY for {agent.name}:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Best Episode: {np.max(episode_rewards):.2f}")
    print(f"  Worst Episode: {np.min(episode_rewards):.2f}")
    print(f"  Success Rate: {sum(1 for r in episode_rewards if r >= 200) / n_episodes * 100:.1f}%")
    print(f"{'='*70}\n")


def compare_agents_side_by_side(agent1, agent2, n_episodes=1):
    """
    Run both agents on the same episode seeds for fair comparison.
    
    Args:
        agent1: First TrainedAgent
        agent2: Second TrainedAgent
        n_episodes: Number of episodes to compare
    """
    print(f"\n{'='*70}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{agent1.name} vs {agent2.name}")
    print(f"{'='*70}\n")
    
    results1 = []
    results2 = []
    
    for episode in range(n_episodes):
        seed = 42 + episode  # Fixed seeds for reproducibility
        
        print(f"\n--- Episode {episode + 1} (Seed: {seed}) ---\n")
        
        # Run agent 1
        env = gym.make('LunarLander-v3')
        state_full, _ = env.reset(seed=seed)
        state = discretize_state(state_full)
        
        reward1 = 0
        steps1 = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent1.choose_action(state)
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            state = next_state
            reward1 += reward
            steps1 += 1
        
        env.close()
        results1.append(reward1)
        
        # Run agent 2 on same seed
        env = gym.make('LunarLander-v3')
        state_full, _ = env.reset(seed=seed)
        state = discretize_state(state_full)
        
        reward2 = 0
        steps2 = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent2.choose_action(state)
            next_state_full, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_full)
            state = next_state
            reward2 += reward
            steps2 += 1
        
        env.close()
        results2.append(reward2)
        
        # Episode comparison
        print(f"{agent1.name}:")
        print(f"  Reward: {reward1:.2f} | Steps: {steps1}")
        print(f"{agent2.name}:")
        print(f"  Reward: {reward2:.2f} | Steps: {steps2}")
        
        winner = agent1.name if reward1 > reward2 else agent2.name if reward2 > reward1 else "Tie"
        print(f"  → Winner: {winner}")
    
    # Overall comparison
    print(f"\n{'='*70}")
    print(f"OVERALL COMPARISON:")
    print(f"  {agent1.name}: Mean = {np.mean(results1):.2f}, Std = {np.std(results1):.2f}")
    print(f"  {agent2.name}: Mean = {np.mean(results2):.2f}, Std = {np.std(results2):.2f}")
    
    if np.mean(results1) > np.mean(results2):
        print(f"  🏆 {agent1.name} wins overall!")
    elif np.mean(results2) > np.mean(results1):
        print(f"  🏆 {agent2.name} wins overall!")
    else:
        print(f"  🤝 It's a tie!")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration function."""
    
    print("\n" + "="*70)
    print("LUNAR LANDER - TRAINED AGENT DEMONSTRATION (v3)")
    print("Agents: Q-Learning, SARSA, Monte Carlo")
    print("="*70)
    
    # Try to load both agents
    q_learning_path = 'results/v1/q_learning_model.pkl'
    sarsa_path = 'results/v1/sarsa_model.pkl'
    
    agents = []
    
    try:
        agent_q = TrainedAgent(q_learning_path, name="Q-Learning")
        agents.append(agent_q)
    except FileNotFoundError:
        print(f"⚠ Q-Learning model not found at: {q_learning_path}")
        agent_q = None
    
    try:
        agent_sarsa = TrainedAgent(sarsa_path, name="SARSA")
        agents.append(agent_sarsa)
    except FileNotFoundError:
        print(f"⚠ SARSA model not found at: {sarsa_path}")
        agent_sarsa = None
    
    
    if not agents:
        print("\n❌ No trained models found!")
        print("Please run the training script first to generate model files.")
        return
    
    print(f"\n✓ Loaded {len(agents)} trained agent(s)\n")
    
    # Interactive menu
    while True:
        print("\n" + "="*70)
        print("DEMONSTRATION OPTIONS:")
        print("="*70)
        
        if agent_q:
            print("  1. Watch Q-Learning agent (3 episodes)")
            print("  2. Watch Q-Learning agent (1 episode, slow motion)")
        
        if agent_sarsa:
            print("  3. Watch SARSA agent (3 episodes)")
            print("  4. Watch SARSA agent (1 episode, slow motion)")
        
        # Video recording options
        print("\n  --- Video Recording ---")
        print("  5. Record video of an agent")
        print("  6. Record comparison videos of all agents")
        
        if len(agents) >= 2:
            print("\n  --- Comparisons ---")
            print("  7. Compare two agents (same random seeds)")
            print("  8. Watch all agents sequentially")
        
        print("\n  0. Exit")
        print("="*70)
        
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '0':
                print("\nExiting demonstration. Goodbye!")
                break
            
            elif choice == '1' and agent_q:
                demonstrate_agent(agent_q, n_episodes=5, delay=0.01)
                #evaluate_agent(agent_q, n_episodes=5, render=True)
            
            elif choice == '2' and agent_q:
                print("\n🎬 Slow motion mode - watch carefully!")
                demonstrate_agent(agent_q, n_episodes=1, delay=0.03)
            
            elif choice == '3' and agent_sarsa:
                demonstrate_agent(agent_sarsa, n_episodes=3, delay=0.01)
            
            elif choice == '4' and agent_sarsa:
                print("\n🎬 Slow motion mode - watch carefully!")
                demonstrate_agent(agent_sarsa, n_episodes=1, delay=0.03)
            
            elif choice == '5':
                # Record video of selected agent
                print("\nAvailable agents:")
                for i, agent in enumerate(agents, 1):
                    print(f"  {i}. {agent.name}")
                
                try:
                    agent_idx = int(input("Select agent (number): ")) - 1
                    if 0 <= agent_idx < len(agents):
                        n_eps = int(input("How many episodes to record? (1-10): "))
                        n_eps = max(1, min(10, n_eps))
                        video_folder = input("Video folder (default: 'videos'): ").strip() or "videos"
                        record_agent_video(agents[agent_idx], n_episodes=n_eps, video_folder=video_folder)
                    else:
                        print("❌ Invalid agent selection.")
                except ValueError:
                    print("❌ Invalid input.")
            
            elif choice == '6':
                # Record comparison videos
                try:
                    n_eps = int(input("Episodes per agent (1-10): "))
                    n_eps = max(1, min(10, n_eps))
                    video_folder = input("Video folder (default: 'videos'): ").strip() or "videos"
                    record_comparison_video(agents, n_episodes=n_eps, video_folder=video_folder)
                except ValueError:
                    print("❌ Invalid input.")
            
            elif choice == '7' and len(agents) >= 2:
                print("\nAvailable agents:")
                for i, agent in enumerate(agents, 1):
                    print(f"  {i}. {agent.name}")
                
                try:
                    agent1_idx = int(input("Select first agent (number): ")) - 1
                    agent2_idx = int(input("Select second agent (number): ")) - 1
                    
                    if 0 <= agent1_idx < len(agents) and 0 <= agent2_idx < len(agents):
                        n_eps = int(input("How many episodes to compare? (1-10): "))
                        n_eps = max(1, min(10, n_eps))
                        compare_agents_side_by_side(agents[agent1_idx], agents[agent2_idx], n_episodes=n_eps)
                    else:
                        print("❌ Invalid agent selection.")
                except ValueError:
                    print("❌ Invalid input.")
            
            elif choice == '8' and len(agents) >= 2:
                for agent in agents:
                    demonstrate_agent(agent, n_episodes=2, delay=0.01)
                    if agent != agents[-1]:
                        input(f"\nPress Enter to continue to {agents[agents.index(agent) + 1].name}...")
            
            else:
                print("❌ Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()