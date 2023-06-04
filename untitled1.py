import random
import numpy as np
import matplotlib.pyplot as plt

def simulate_trajectory(num_steps):
    
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    mu = np.random.uniform(-2, 2)
    v[0] = np.random.normal(0, 1)
    
    for t in range(1, num_steps):
        v[t] = v[t-1] + 0.2 * (mu - v[t-1]) + 0.32 * np.random.normal(0, 1)  
        x[t] = x[t-1] + v[t-1]  

    return x, v, mu

num_steps = 50
x, v, mu = simulate_trajectory(num_steps)

time = np.arange(num_steps)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x, label='Pozicija')
plt.xlabel('vreme')
plt.ylabel('pozicija')

plt.subplot(2, 1, 2)
plt.plot(time, v, label='brzina')
plt.axhline(mu, color='red', linestyle='--', label='mu')
plt.legend()
plt.xlabel('vreme')
plt.ylabel('brzina')

plt.tight_layout()
plt.show()

def generate_observations(x, num_steps):
    
    e = np.zeros(num_steps)
    theta = np.random.binomial(1, 0.05, num_steps)
    
    for t in range(num_steps):
        n = np.random.laplace(scale=np.sqrt(np.abs(x[t])) / 5)
        if theta[t] == 1:
            e[t] = -x[t] + n
        else:
            e[t] = x[t] + n

    return e

obs = generate_observations(x, num_steps)

plt.figure()
plt.plot(time, x, label='tacna pozicija')
plt.plot(time, obs, label='merena pozicija')
plt.xlabel('vreme')
plt.ylabel('pozicija')
plt.legend()
plt.show()


def particle_filter(observations, num_particles, resampling):

    
    particles = np.zeros((num_steps, num_particles))
    velocities = np.zeros((num_steps, num_particles))
    weights = np.zeros((num_steps, num_particles))
    weights[0] = np.ones(num_particles) / num_particles
    mu = np.random.uniform(-2, 2, num_particles)
    estimates = np.zeros(num_steps)
    vestimates = np.zeros(num_steps)
    scatterweigh = []

    def resample(particless, velocitiess, weights, t):
        # retparticles = np.zeros(100)
        # retvelocities = np.zeros(100)
        indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
        # for i in range(100):
        #     retparticles[i] = particless[:, indices[i]]
        #     retvelocities[i] = velocitiess[:, indices[i]]
        return particless[indices], velocitiess[indices]

    for t in range(num_steps):
        if t == 0:
            velocities[t] = np.random.normal(0, 1, num_particles)
            particles[t] = np.zeros(num_particles)  
            estimates[t] = 0
            vestimates[t] = v[0]
        else:
            velocities[t] = velocities[t - 1] + 0.2*(mu - velocities[t - 1]) + 0.32 * np.random.normal(0, 1, num_particles)
            particles[t] = particles[t - 1] + velocities[t - 1]
            b = np.sqrt(np.abs(particles[t])) / 5  
            p_n = np.exp(-np.abs(observations[t] - particles[t]) / b) / (2 * b)
            if t == 0:
                pass
            else:
                weights[t] = weights[t - 1]*p_n
                weights[t] /= np.sum(weights[t])  
                estimates[t] = np.dot(weights[t], particles[t])
                vestimates[t] = np.dot(weights[t], velocities[t])
            if resampling == 1:
                continue
            elif resampling == 2:
                particlesres, velocitiesres = resample(particles[t], velocities[t], weights[t], t)
                particles[t] = particlesres
                velocities[t] = velocitiesres
                weights[t] = np.ones(num_particles) / num_particles      
            elif resampling == 3:
                if 1 / np.sum(weights[t] ** 2) < num_particles / 2:
                    particlesres, velocitiesres = resample(particles[t], velocities[t], weights[t], t)
                    particles[t] = particlesres
                    velocities[t] = velocitiesres
                    weights[t] = np.ones(num_particles) / num_particles
            else:
                if t == 9 or t == 19 or t == 29 or t == 39:
                    scatterweigh.append(weights[t])
                elif t == 10 or t == 20 or t == 30 or t == 40:
                    particlesres, velocitiesres = resample(particles[t], velocities[t], weights[t], t)
                    particles[t] = particlesres
                    velocities[t] = velocitiesres
                    weights[t] = np.ones(num_particles) / num_particles                
                    
    return particles, estimates, vestimates, weights, scatterweigh

num_particles = 1000
estimated_particles, estimates, vestimates, weights, sweight = particle_filter(obs, num_particles, resampling=4)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x, label='tacna pozicija')
plt.plot(time, obs, 'r.', label='izmerena pozicija')
plt.plot(time, estimates, 'g', label='estimirana pozicija')
plt.legend()
plt.xlabel('vreme')
plt.ylabel('pozicija')

plt.subplot(2, 1, 2)
plt.plot(time, v, label='tacna brzina')
plt.plot(time, vestimates, 'g', label='estimirana brzina')
plt.legend()
plt.xlabel('vreme')
plt.ylabel('brzina')

plt.tight_layout()
plt.show()

def visualize_particles(particles, weights, time_index):
    
    plt.scatter(time_index*np.ones(num_particles), particles[time_index], s=weights[time_index] * 1000, alpha=0.5, label='Particles')
    plt.axhline(x[time_index], color='r', linestyle='--', label='Tacna pozicija')
    plt.axhline(obs[time_index], color='g', linestyle='--', label='Izmerena pozicija')
    plt.xlabel('vreme')
    plt.ylabel('pozicija')


plt.figure(figsize=(8, 6))
time_indices = [9, 19, 29, 39]  # Time points to visualize particles
for t in time_indices:
    visualize_particles(estimated_particles, weights, t)
plt.show()

#%%
def calculate_rmse(estimated_particles, x):
    rmse = np.sqrt(np.mean((estimated_particles-x)**2))
    return rmse

num_simulations = 100
rmse_no_resampling = np.zeros(num_simulations)
rmse_resampling_every = np.zeros(num_simulations)
rmse_conditional_resampling = np.zeros(num_simulations)

for i in range(num_simulations):
    
    x, v, mu = simulate_trajectory(num_steps)
    obs = generate_observations(x, num_steps)
    a, no_resampling, b, c, d = particle_filter(obs, num_particles, resampling=1)
    a, resampling_every, b, c, d = particle_filter(obs, num_particles, resampling=2)
    a, conditional_resampling, b, c, d = particle_filter(obs, num_particles, resampling=3)

    rmse_no_resampling[i] = calculate_rmse(no_resampling, x)
    rmse_resampling_every[i] = calculate_rmse(resampling_every, x)
    rmse_conditional_resampling[i] = calculate_rmse(conditional_resampling, x)

rmse_no_resampling = np.sort(rmse_no_resampling)
rmse_resampling_every = np.sort(rmse_resampling_every)
rmse_conditional_resampling = np.sort(rmse_conditional_resampling)

mean_no_resampling = np.mean(rmse_no_resampling)
mean_resampling_every = np.mean(rmse_resampling_every)
mean_conditional_resampling = np.mean(rmse_conditional_resampling)

std_no_resampling = np.std(rmse_no_resampling)
std_resampling_every = np.std(rmse_resampling_every)
std_conditional_resampling = np.std(rmse_conditional_resampling)



fig, ax = plt.subplots()

plt.ylabel('RMSE')
vp = ax.violinplot([rmse_no_resampling, rmse_resampling_every, rmse_conditional_resampling],
                   [2, 4, 6], widths=2,
                   showmedians=True, showextrema=True)

for body in vp['bodies']:
    body.set_alpha(0.6)

# plt.figure(figsize=(5,12))
# plt.subplot(311)
print("no mean", mean_no_resampling)
print("no std", std_no_resampling)
# plt.hist(rmse_no_resampling)
# plt.
# plt.subplot(312)
print("every mean", mean_resampling_every)
print("every std", std_resampling_every)
# plt.hist(rmse_resampling_every)
# plt.subplot(313)
print("con mean", mean_conditional_resampling)
print("con std", std_conditional_resampling)
# plt.hist(rmse_conditional_resampling)







