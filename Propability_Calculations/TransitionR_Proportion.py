import numpy as np

# Define transition rates
transition_rates = {
    '1 to 1': 48.28,
    '1 to 2': 0.08,
    '2 to 1': 0.27,
    '2 to 2': 36.0
}

# Simulation parameters
total_time = 10000 # seconds
time_step = 0.1  # seconds

# Initialize counters for time spent in each behavior
time_in_behavior_1 = 0
time_in_behavior_2 = 0

# Randomly choose the starting behavior
behavior = np.random.choice([1, 2])

# Simulate behavior over 1000 seconds
time = 0
while time < total_time:
    # Calculate transition probabilities based on transition rates
    p_1_to_1 = transition_rates['1 to 1'] / (transition_rates['1 to 1'] + transition_rates['1 to 2'])
    p_1_to_2 = transition_rates['1 to 2'] / (transition_rates['1 to 1'] + transition_rates['1 to 2'])
    p_2_to_1 = transition_rates['2 to 1'] / (transition_rates['2 to 1'] + transition_rates['2 to 2'])
    p_2_to_2 = transition_rates['2 to 2'] / (transition_rates['2 to 1'] + transition_rates['2 to 2'])

    # Determine next behavior based on transition probabilities
    if behavior == 1:
        next_behavior = np.random.choice([1, 2], p=[p_1_to_1, p_1_to_2])
    else:
        next_behavior = np.random.choice([1, 2], p=[p_2_to_1, p_2_to_2])

    # Update time and behavior
    time += time_step
    behavior = next_behavior

    # Update time spent in each behavior
    if behavior == 1:
        time_in_behavior_1 += time_step
    else:
        time_in_behavior_2 += time_step

# Calculate percentage of time spent in behavior 2
percentage_in_behavior_2 = (time_in_behavior_2 / total_time) * 100
print(f"Percentage of time spent in behavior 2: {percentage_in_behavior_2:.2f}%")
