import re
import argparse
import matplotlib.pyplot as plt

def parse_log(log_file):
    steps = []
    rewards = []
    losses = []
    gen_times = []
    train_times = []
    weight_sync_times = []
    total_times = []
    
    # Regexes for robust parsing
    sync_pattern = re.compile(r"Weight sync:.*total=(?P<time>[-+]?\d*\.\d+)s")
    step_pattern = re.compile(r"Step\s+(?P<step>\d+)")
    loss_pattern = re.compile(r"Loss:\s*(?P<loss>[-+]?\d*\.\d+)")
    reward_pattern = re.compile(r"Reward:\s*(?P<reward>[-+]?\d*\.\d+)")
    total_time_pattern = re.compile(r"\|\s*Time:\s*(?P<time>[-+]?\d*\.\d+)s")
    gen_time_pattern = re.compile(r"Gen Time:\s*(?P<time>[-+]?\d*\.\d+)s")
    train_time_pattern = re.compile(r"Train Time:\s*(?P<time>[-+]?\d*\.\d+)s")
    
    latest_weight_sync = 0.0
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for weight sync log line
            sync_match = sync_pattern.search(line)
            if sync_match:
                latest_weight_sync = float(sync_match.group('time'))
                continue
                
            # Check for main step metrics log line
            step_match = step_pattern.search(line)
            if step_match:
                loss_match = loss_pattern.search(line)
                reward_match = reward_pattern.search(line)
                gen_match = gen_time_pattern.search(line)
                train_match = train_time_pattern.search(line)
                total_match = total_time_pattern.search(line)
                
                # Make sure we have the critical fields
                if loss_match and reward_match:
                    steps.append(int(step_match.group('step')))
                    losses.append(float(loss_match.group('loss')))
                    rewards.append(float(reward_match.group('reward')))
                    
                    # Optional/new fields
                    gen_times.append(float(gen_match.group('time')) if gen_match else 0.0)
                    train_times.append(float(train_match.group('time')) if train_match else 0.0)
                    total_times.append(float(total_match.group('time')) if total_match else 0.0)
                    weight_sync_times.append(latest_weight_sync)
                    
                    # Reset weight sync for next step
                    latest_weight_sync = 0.0
                    
    return steps, rewards, losses, gen_times, train_times, weight_sync_times, total_times

def plot_metrics(log_file, output_image):
    steps, rewards, losses, gen_times, train_times, weight_sync_times, total_times = parse_log(log_file)
    
    if not steps:
        print("No metrics found in the log file.")
        return

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot Reward
    ax1.plot(steps, rewards, marker='o', color='blue', label='Reward')
    ax1.set_title('GRPO Reward over Steps')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Loss
    ax2.plot(steps, losses, marker='o', color='red', label='Loss')
    ax2.set_title('GRPO Loss over Steps')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Step Times
    ax3.plot(steps, gen_times, marker='o', color='green', label='Sampler (Gen) Time')
    ax3.plot(steps, train_times, marker='o', color='orange', label='Trainer (Fwd/Bwd/Opt) Time')
    if any(weight_sync_times):
        ax3.plot(steps, weight_sync_times, marker='o', color='purple', label='Weight Transfer (Sync) Time')
    if any(total_times):
        ax3.plot(steps, total_times, marker='o', color='brown', label='Total Step Time')
    ax3.set_title('Step Time Breakdown')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Time (seconds)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot successfully saved to {output_image}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RL training metrics from log file")
    parser.add_argument("--log_file", type=str, default="training_200.log", help="Path to the training log file")
    parser.add_argument("--output", type=str, default="rl_metrics.png", help="Path to save the output plot")
    args = parser.parse_args()
    
    plot_metrics(args.log_file, args.output)
