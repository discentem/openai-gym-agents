import tensorflow as tf
import gym
import re
import numpy as np

env = gym.make('CartPole-v0')
n_input = env.observation_space.shape[0]
n_actions = int(re.findall('\d',str(env.action_space))[0])

# Parameters
learning_rate = 1.e-1
training_epochs = 2000
n_steps = 100
display_step = 100
exp_a = 0.1
exp_b = 0.0
batch_size = 100
loss_multiplier = 1
max_memory = 20000
discount = 0.9

# Network Parameters
n_hidden_1 = 4 # 1st layer number of features
n_hidden_2 = 4 # 2nd layer number of features

# tf Graph input
x = tf.placeholder(tf.float64, [None, n_input])
y = tf.placeholder(tf.float64, [None, n_actions])

class ExperienceQModel(object):
    def __init__(self, max_memory=500, discount=.9):
        # Memory replay parameters
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def exp_remember(self, states):
        self.memory.append(states.copy())
        if len(self.memory) > self.max_memory:
          del self.memory[0]

    # based on https://gist.github.com/EderSantana/c7222daa328f0e885093
    def exp_get_batch(self,batch_size=10):
        len_memory = len(self.memory)
        n_examples = min(len_memory, batch_size)
        inputs = np.zeros((n_examples, n_input))
        targets = np.zeros((n_examples, n_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,size=n_examples)):
            #get_memory
            states = self.memory[idx]
            state_t = states['state_t']
            state_tp1 = states['state_tp1']
            action = states['action']

            # input
            inputs[i] = state_t.astype(np.float64)

            # targets - not correcting those which are not taken
            feed_dict = {x: states['state_t'].reshape(1,-1)}
            targets[i] = sess.run(pred, feed_dict)
            
            # acted action
            feed_dict = {x: states['state_tp1'].reshape(1,-1)}
            Qsa = np.max(sess.run(pred, feed_dict))

            # check if endgame and if not apply discount
            if states['endgame']:
                targets[i,action] = states['reward'] # assign just reward if endgame
            else:
                targets[i,action] = states['reward'] + self.discount * Qsa
        return inputs, targets


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Hidden layer with RELU activation
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],dtype=tf.float64)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],dtype=tf.float64)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_actions],dtype=tf.float64))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],dtype=tf.float64)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2],dtype=tf.float64)),
    'out': tf.Variable(tf.random_normal([n_actions],dtype=tf.float64))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Launch the graph
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# initialize states and experience replay
states = {}
exp_replay = ExperienceQModel(max_memory=max_memory)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    state_tp1 = env.reset()
    done = False

    for t in range(n_steps):
        env.render()
        state_t1 = np.array(state_tp1)
        
        # exploration cycle
        if np.random.rand() <= exp_a-exp_b*epoch/100:
            action = env.action_space.sample()
        else:
            feed_dict = {x: state_t1.reshape(1,-1)}
            qvals = sess.run(pred, feed_dict)
            action = np.argmax(qvals)

        # take a next step
        state_tp1, reward, done, info = env.step(action)

        # rewards
        if (t == 99) and (done == False):
            print("{}: won!".format(epoch))
        if reward < 0:
            reward = -1.0*loss_multiplier;

        # store experience
        states['action'] = action
        states['reward'] = float(reward)
        states['endgame'] = done
        states['state_t'] = np.array(state_t1)
        states['state_tp1'] = np.array(state_tp1)
        exp_replay.exp_remember(states)

        # get experience replay
        x_batch, y_batch = exp_replay.exp_get_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
        # Compute average loss
        avg_cost += c / n_steps

        # Lost
        if done:
            print("{}: lost after {}, cost {}".format(epoch,t+1,avg_cost))
            break