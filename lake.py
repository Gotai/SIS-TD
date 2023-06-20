import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import gymnasium as gym

# For reproduction
gym.utils.seeding.np_random(42)

# Example for the Q-Table discussed at length in the Report
#env = gym.make('FrozenLake-v1', desc=["SH", "FG"], is_slippery=False , render_mode="rgb_array")

# Example for the 4x4 environment discussed in the performance part of Q-Learning
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False , render_mode="rgb_array")



# States will be tuples as we are looking at NxN Matrices for our environment

class LakeRunner:
    def __init__(self, learning_rate, epsilon, decay, final_epsilon, gamma = .9, q_table=''):
        # create Q-Table, we need a (B, T)
        if q_table != '':
            try:
                self.q_tabel = np.genfromtxt(q_table, delimiter=',')
            except:
                self.q_tabel = np.zeros([env.observation_space.n, env.action_space.n])
        else:
            self.q_tabel = np.zeros([env.observation_space.n, env.action_space.n])        
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.final_epsilon = final_epsilon
        self.delta = []


    def greedy_pi(self, state):
        # we return the entire row vector of the actions
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_tabel[state]))
    
    # note that reward is actually reward_t+1
    def update(self, state, action, reward, next_state):
        # Splitting this into error and target, so we can track the target

        error =  reward + self.gamma * np.max(self.q_tabel[next_state]) - self.q_tabel[state][action]
        self.q_tabel[state][action] = self.q_tabel[state][action] + self.lr * error
        self.delta.append(error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.decay)

    def forwadpass(self):
        state, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = int(np.argmax(self.q_tabel[state]))
            next_state, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            plt.imshow(frame)
            plt.show()
            # update if the environment is done and the current state
            done = terminated or truncated
            state = next_state

# hyperparameter
learning_rate = 0.01
max_iter = 200000
epsilon = 1.0
decay = epsilon / (max_iter/ 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = LakeRunner(learning_rate, epsilon, decay, final_epsilon)

env.reset()
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=max_iter)
sequenz = []

def training(max_iter):
    for episode in tqdm(range(max_iter)):
        state, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.greedy_pi(state)
            sequenz.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # update the agent
            agent.update(state, action, reward, next_state)
            #frame = env.render()
            #plt.imshow(frame)
            #plt.show()
            # update if the environment is done and the current state
            done = terminated or truncated
            state = next_state

        agent.decay_epsilon()
    np.savetxt("agent.csv", agent.q_tabel, delimiter=",")

training(max_iter)
print("Success rate: " + str(sum(env.return_queue)/len(env.return_queue)))

agent.forwadpass()

# Get Latex Graphs
matplotlib.use("pgf")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# page width 404.02914pt.
# Width of figure (in pts)
#
# curtisy of https://jwalton.info/Matplotlib-latex-PGF/ 20.06.2023
def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

width , height = set_size(405, fraction=1)

# fig1, axs1 = plt.subplots()
# axs1.set_title("Episode rewards")
# plt.tight_layout()
# fig1.set_size_inches(w=5, h=4)
# avg_reward = (env.return_queue)/
# axs1.plot(range(len(env.return_queue)), env.return_queue)
# plt.show()

# fig2, axs2 = plt.subplots()
# axs2.set_title("Episode Length")
# plt.tight_layout()
# fig2.set_size_inches(w=5, h=4)
# axs2.plot(range(len(env.length_queue)), env.length_queue)
# plt.show()

# fig3, axs3 = plt.subplots()
# axs3.set_title("Training Error")
# plt.tight_layout()
# fig3.set_size_inches(w=5, h=4)
# axs3.plot(range(len(agent.delta)), agent.delta)
# plt.show()

# rolling_length = 20000
# fig1, axs1 = plt.subplots()
# axs1.set_title("Episode rewards")
# reward_moving_average = (
#     np.convolve(
#         np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# axs1.plot(range(len(reward_moving_average)), reward_moving_average)
# fig1.tight_layout()
# fig1.set_size_inches(w=width, h=height)
# plt.savefig('episode_reward_slip.pgf')
# #plt.savefig('episode_reward.pgf')

# fig2, axs2 = plt.subplots()
# axs2.set_title("Episode lengths")
# length_moving_average = (
#     np.convolve(
#         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#     )
#     / rolling_length
# )
# axs2.plot(range(len(length_moving_average)), length_moving_average)
# fig2.tight_layout()
# fig2.set_size_inches(w=width, h=height)
# plt.savefig('episode_length_slip.pgf')
# #plt.savefig('episode_length.pgf')

# fig3, axs3 = plt.subplots()
# axs3.set_title("Training Error")
# training_error_moving_average = (
#     np.convolve(np.array(agent.delta), np.ones(rolling_length), mode="same")
#     / rolling_length
# )
# axs3.plot(range(len(training_error_moving_average)), training_error_moving_average)
# fig3.set_size_inches(w=width, h=height)
# fig3.tight_layout()
# plt.savefig('train_error_slip.pgf')
#plt.savefig('train_error.pgf')

rolling_length = int(max_iter//10)
fig, axs = plt.subplots(ncols=3, figsize=(width, height))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)


axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.delta), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
#plt.set_size_inches(w=width, h=height)
plt.tight_layout()
plt.savefig('train_data.pgf')
