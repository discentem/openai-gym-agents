import gym
import re

class ExperienceQModel(object):
	def __init__(self, env='CartPole-v0', max_memory=500, discount=.9):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount

		self.env = gym.make(env)
		self.n_obs = self.env.observation_space.shape[0]
		self.n_act = self.env.action_space.shape[0]

	def exp_remember(self, states):
		self.memory.append(states.copy())
		if len(self.memory) > self.max_memory:
			del self.memory[0]

	def exp_get_batch(self, model, batch_size=10):
		len_memory = len(self.memory)
		n_examples = min(len_memory, batch_size)
		inputs = np.zeros((n_examples, n_features))
		targets = np.zeros((n_examples, n_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory,size=n_examples)):
			
			#get_memory
			states = self.memory[idx]
			state_t = states['state_t'].reshape(1,-1)
			state_tp1 = states['state_tp1'].reshape(1,-1)
			
			inputs[i:i+1] = state_t # assign features

			if states['endstep']:
				targets[i] = states['reward'] # assign reward in the end step
			else:
				Q_sa = np.max(model.predict(state_tp1)[0])
				targets[i] = states['reward'] + self.discount * Q_sa
		return inputs, targets


# Gym Model object
class GymModel(object):
	def __init__(self,exp,env='CartPole-v0'):
		self.env = gym.make('CartPole-v0')
		self.exp = ExperienceReplay(max_memory=max_memory, discount=discount)
		self.n_obs = self.env.observation_space.shape[0]
		self.n_act = int(re.findall('\d',str(self.env.action_space))[0])

	def train(n_episodes=20,n_steps=100,n_batch=50)
		# Get initial info about environment
		n_obs = env.observation_space
		actionenv.action_space

		
		# Create placeholders for features and labels
		env_placeholder = tf.placeholder(tf.float32, shape=(n_batch,self.n_obs))
		tgt_placeholder = tf.placeholder(tf.int32, shape=(n_batch,self.n_act))

		for i_episode in range(n_episodes):
    		observation = env.reset()



    for t in range(n_steps):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break



if __name__ == "__main__":
	n_episodes = 20
	n_steps = 100
	n_batch = 50

	# experience replay
	exp = ExperienceReplay(max_memory=max_memory, discount=discount)

	# environment
	gym = GymModel(exp=exp, env='CartPole-v0')

	# train the model
	gym.train()