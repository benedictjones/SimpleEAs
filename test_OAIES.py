import numpy as np
from pyeas._oaies import OAIES
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import matplotlib
# matplotlib.use("Agg")

def q2(x1, x2):
    return (x1 + 3) + (10 * (x2 + 2)) ** 2

def q3(x1, x2, x3):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2 + (x3**3)

def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5

def rmse(x, y, w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))

x = np.linspace(0, 10, 500)
rng = np.random.default_rng(0)
y = np.cos(x) # + rng.normal(0, 0.2, 500)

optimizer = OAIES(
                alpha=0.01,
                sigma=0.001,
                bounds=np.array([[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5]]),
                #bounds=np.array([[-5,5],[-5,5]]),
                #groupings=[2,4],
                population_size=20,
                optimiser = 'adam',  # 'ttb1', rand1
                seed=1)

num_gens = 400
for generation in range(num_gens):
    # print("Gen:", generation)
    solutions = []
    
    # Ask a parameter
    trial_pop = optimizer.ask(loop=generation)
    # print(trial_pop)

    for j, trial in enumerate(trial_pop):
        #value = fmodel(x, trial)
        # print(j)
        value = rmse(x, y, trial)

        # value = q3(trial[0], trial[1], trial[2])
        solutions.append((value))
        #print(f"#{generation} {value} (x1={trial[0]}, x2 = {trial[1]}))")
        #print(f"#{generation} {value} (x1={trial[0]}, x2 = {trial[1]}), , x3 = {trial[2]})")

    # Tell evaluation values.
    optimizer.tell(solutions, trial_pop, t=generation)

    # Calc the new parent fitness, and Tell Again!
    parent_fit = rmse(x, y, optimizer.parent)
    optimizer.tellAgain(parent_fit)

    print("Gen:", generation, optimizer.best[0], " best trial fit:", optimizer.best_trial[0])

fig = plt.figure()
plt.plot(optimizer.history['best_fits'])
plt.yscale("log")


print(optimizer.history['best_solutions'][-1])

fig, ax = plt.subplots()
ax.scatter(x, y, marker=".", color='r', alpha=0.7, label='Target data')
plt.plot(x, np.cos(x), '--', label='ideal cos(x)', color='k', alpha=0.5)
data = fmodel(x, optimizer.history['best_solutions'][-1])
ax.plot(x, data, label='OAIES Solution')
ax.legend()


plt.show()
exit()

fig_ani, (ax, ax2) = plt.subplots(ncols=2, figsize=(9,4))

ax.set_ylim([-10, 10])
ax.scatter(x, y, marker=".", color='r')
ax.set_xlabel("x")
ax.set_ylabel("y")

ax2.set_yscale('log')
ax2.plot(optimizer.history['best_fits'])
ax2.set_xlabel("Generation")
ax2.set_ylabel("rmse")
it_line, = ax2.plot([0, 0],  [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], markersize=5, color='k', alpha=0.5) #
plt.tight_layout()

lines = []


def ani(i):
    # ax.clear()

    lim = 10 - (i/num_gens)*(10-4)

    ax.set_ylim([-lim, lim])

    data = fmodel(x, optimizer.history['best_solutions'][i])
    line, = ax.plot(x, data, alpha=0.4)

    lines.append(line)
    if len(lines) > 10:
        lines[0].remove()
        lines.pop(0)

    # ax.set_title("Generation %d" % (i))

    it_line.set_xdata([i, i])

FPS = num_gens/300
the_animation = animation.FuncAnimation(fig_ani, ani, frames=np.arange(num_gens), interval=20)

fig_path = "example_OAIES.gif"
the_animation.save(fig_path, writer='pillow', fps=FPS, dpi=100)


plt.show()