import numpy as np
import matplotlib.pyplot as plt
from scenarioGroundStripImaging import run
from tqdm import tqdm
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


# CE optimization parameters
n_samples = 500
n_iterations = 10
n_elite = int(0.1*n_samples)

# define the parameters to optimiza with CE
parameters_bounds = np.array([[1,50],[-0.01,0.01],[10,200],[0,6]])
parameters_distributions = [] # this is a list of dictionaries, each dictionary is a parameter distribution, by default it has the mean and std for a normal distribution
n_parameters = parameters_bounds.shape[0]
best_parameters = []
best_parameters_fitness = []

# Start optimization loop
for i in tqdm(range(n_iterations), desc="CE Optimization", unit="iteration"):
    # Generate random samples. First iteration is uniform, the rest are normal distributed
    if i == 0:
        samples = np.random.uniform(parameters_bounds[:,0], parameters_bounds[:,1], (n_samples, n_parameters))
    else:
        means = parameters_distributions[-1]['mean']
        stds = parameters_distributions[-1]['std']
        samples = np.random.normal(means, stds, (n_samples, n_parameters))
        # Clip samples to the bounds
        samples = np.clip(samples, parameters_bounds[:,0], parameters_bounds[:,1])

    # Evaluate samples
    fitness = np.zeros(n_samples)
    for j in tqdm(range(n_samples), desc=f"Evaluating Samples (Iteration {i})", leave=False):
        _, fitness[j] = run(show_plots=False, parameters=samples[j])
   
   # Sort samples
    idx = np.argsort(fitness)
    sorted_samples = samples[idx]
    elite_samples = sorted_samples[:n_elite]
    # Update parameters distribution
    parameters_distribution = {'mean': np.mean(elite_samples, axis=0), 'std': np.std(elite_samples, axis=0)}
    parameters_distributions.append(parameters_distribution)
    # save best parameters
    best_parameters.append(elite_samples[0])
    best_parameters_fitness.append(fitness[idx[0]])
    # Print best sample
    # print(f'Iteration {i}: Best sample: {elite_samples[0]}, Fitness: {fitness[idx[0]]}')
    tqdm.write(f'Iteration {i}: Best sample: {elite_samples[0]}, Fitness: {fitness[idx[0]]}')

## Plots
# Convert lists to numpy arrays for plotting
best_parameters = np.array(best_parameters)

# Plot evolution of best fitness
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(best_parameters_fitness)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Fitness')
ax1.set_title('Best Fitness Evolution')
ax1.grid()
plt.tight_layout()
plt.show()

# Plot evolution of mean, std, and best sample value for each parameter
for param_idx in range(n_parameters):
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [dist['mean'][param_idx] for dist in parameters_distributions]
    stds = [dist['std'][param_idx] for dist in parameters_distributions]
    best_values = best_parameters[:, param_idx]
    
    ax.plot(means, label='Mean')
    ax.fill_between(range(n_iterations), 
                    np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    alpha=0.2, label='Std Dev')
    ax.plot(best_values, label='Best Sample Value', linestyle='--')
    
    # Add horizontal lines for parameter bounds
    lower_bound, upper_bound = parameters_bounds[param_idx]
    ax.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
    ax.axhline(y=upper_bound, color='g', linestyle='--', label='Upper Bound')
    
    # Calculate final mean value
    final_mean = means[-1]
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Parameter {param_idx + 1} Value')
    ax.set_title(f'Evolution of Parameter {param_idx + 1} (Final Mean: {final_mean})')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

# Run the final best parameters with the plots
_, fitness = run(show_plots=True, parameters=best_parameters[-1])