import numpy as np

def rates_to_probabilities(rate_matrix, time_interval):
    prob_matrix = {}
    for key, rate in rate_matrix.items():
        prob_matrix[key] = 1 - np.exp(-rate * time_interval)
    return prob_matrix

# Define transition rates
transition_rates = {
    '1 to 1': 58.43045,
    '1 to 2': 0.170814824,
    '2 to 1': 0.875374476,
    '2 to 2': 37.59302
}
time_interval = 0.1  # seconds

# Calculate transition probabilities
transition_probabilities = rates_to_probabilities(transition_rates, time_interval)

# Output transition probabilities
print("Transition probabilities:")
for key, prob in transition_probabilities.items():
    print(f"{key}: {prob:.5f}")
