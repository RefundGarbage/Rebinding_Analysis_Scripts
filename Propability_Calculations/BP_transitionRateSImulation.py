import numpy as np

def probabilities_to_rates(prob_matrix, time_interval):
    rate_matrix = {}
    for key, prob in prob_matrix.items():
        rate_matrix[key] = -np.log(1 - prob) / time_interval
    return rate_matrix
#1.2
#TetR
#bound_dif = 0.054713313896987366
#dif_bound = 0.01601317957166392

#     wt
#bound_dif = 0.035961922670114
#dif_bound = 0.01555709205847491

#     dm
#bound_dif = 0.0247050777910754
#dif_bound = 0.00664822520281736

#1.0
#TetR
#bound_dif = 0.05468685364441135
#dif_bound = 0.011766473947559185

#     wt
#bound_dif = 0.03401592718998862
#dif_bound = 0.012287334593572778

#     dm
#bound_dif = 0.037267080745341616
#dif_bound = 0.007412197791264551

#1.2_ .15difProp
#     wt
#bound_dif =  0.03576891398988116
#dif_bound =  0.020731222678500465

#     dm
#bound_dif = 0.024350510160336104
#dif_bound = 0.008955033621211639

#1.2_ .08difProp
#     wt
#bound_dif =  0.035961922670114
#dif_bound =  0.01730864333205121

#     dm
#bound_dif = 0.024630120584965363
#dif_bound = 0.007072311280956871



# Define transition probabilities and time interval
transition_probabilities = {
    '1 to 1': 1-dif_bound,
    '1 to 2': dif_bound,
    '2 to 1': bound_dif,
    '2 to 2': 1-bound_dif
}
time_interval = 0.1  # seconds

# Calculate transition rates
transition_rates = probabilities_to_rates(transition_probabilities, time_interval)

# Output transition rates
print("transition_rates = {")
for key, rate in transition_rates.items():
    print(f"    '{key}': {rate:.5f},")
print("}")

print("[")
print(f"    [{transition_rates['1 to 1']}, {transition_rates['1 to 2']}],")
print(f"    [{transition_rates['2 to 1']}, {transition_rates['2 to 2']}]")
print("]")

# Simulation parameters
total_time = 100000 # seconds
time_step = 0.1  # seconds

# Initialize counters for time spent in each behavior
time_in_behavior_1 = 0
time_in_behavior_2 = 0

# Randomly choose the starting behavior
behavior = np.random.choice([1, 2])

# Simulate behavior
time = 0
while time < total_time:
    # Determine next behavior based on transition rates
    if behavior == 1:
        total_probability = transition_rates['1 to 1'] * time_step + transition_rates['1 to 2'] * time_step
        p_1_to_1 = (transition_rates['1 to 1'] * time_step) / total_probability
        next_behavior = np.random.choice([1, 2], p=[p_1_to_1, 1 - p_1_to_1])
    else:
        total_probability = transition_rates['2 to 1'] * time_step + transition_rates['2 to 2'] * time_step
        p_2_to_1 = (transition_rates['2 to 1'] * time_step) / total_probability
        next_behavior = np.random.choice([1, 2], p=[p_2_to_1, 1 - p_2_to_1])

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
