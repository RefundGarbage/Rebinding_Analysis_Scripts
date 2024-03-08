import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def streak_distribution(n, p):
    probabilities = [0] * (n + 1)  # Initialize probabilities list
    for k in range(1, n+1):
        probability = (1 - p)**(k-1) * p
        probabilities[k] = probability
    return probabilities

def calculate_mean(probabilities):
    mean = 0
    for length, probability in enumerate(probabilities):
        mean += length * probability
    return mean

def mean_streak_length(p, max_length=100000):
    mean = 0
    for k in range(1, max_length + 1):
        probability = (1 - p)**(k-1) * p
        mean += k * probability
    return mean

def find_probability(mean):
    result = minimize_scalar(lambda p: (mean_streak_length(p) - mean)**2, bounds=(0, 1), method='bounded')
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to converge")

n_flips = 10000
p_success = .00001
expected_mean = 1

probabilities = streak_distribution(n_flips, p_success)

plt.bar(range(1, n_flips+1), probabilities[1:])  # Exclude index 0 (no streak)
plt.xlabel('1/10 seconds')
plt.ylabel('Probability')
plt.title('Distribution of Track Lengths')
plt.xlim(0, 100000)  # Adjusting x-axis limit to show only relevant streak lengths
plt.show()

# Calculate the mean of the distribution
mean = calculate_mean(probabilities[1:])
print("Mean based on pSuccess:", mean)

# Example usage
probability = find_probability(expected_mean)
print("Probability of success from expected mean:", probability)
