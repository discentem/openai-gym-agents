import gym
import re
import tensorflow as tf
import numpy as np

class ExperienceQModel(object):
    def __init__(self, env, monitor_file, max_memory=10000, discount=.9, n_episodes=100, 
                 n_steps=100, batch_size=100, learning_rate = 0.01, 
                 exploration=lambda x: 0.1, stop_training=10):
        
        # Memory replay parameters
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

        # exploration
        self.eps = exploration # epsilon-greedy as function of epoch
        
        # environment parameters
        self.env = gym.make(env)
        self.monitor_file = monitor_file
        self.n_input = self.env.observation_space.shape[0]
        self.n_actions = int(re.findall('\d+',str(self.env.action_space))[0]) # shameless hack to get a dim of actions
        
        # training parameters
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.n_steps = n_steps # must be equal to episode length
        self.batch_size = batch_size
        self.stop_training = stop_training # stop training after stop_training consecutive wins
        self.consec_wins = 0 # number of consecutive wins to stop training

        # Neural Network Parameters
        self.n_hidden_1 = 4
        self.n_hidden_2 = 4
        
        # Initialize tensor flow parameters
        self.x = tf.placeholder(tf.float64, [None, self.n_input])
        self.y = tf.placeholder(tf.float64, [None, self.n_actions])
        
        # TF global step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Initialize layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden_1],dtype=tf.float64),name='h1'),
            'h2': tf.Variable(tf.truncated_normal([self.n_hidden_1, self.n_hidden_2],dtype=tf.float64),name='h2'),
            'out': tf.Variable(tf.truncated_normal([self.n_hidden_2, self.n_actions],dtype=tf.float64),name='out')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1],dtype=tf.float64),name='b1'),
            'b2': tf.Variable(tf.zeros([self.n_hidden_2],dtype=tf.float64),name='b2'),
            'out': tf.Variable(tf.zeros([self.n_actions],dtype=tf.float64),name='out')
        }
        
        # define graph
        self.define_model()
        

    def exp_remember(self, states):
        self.memory.append(states.copy())
        if len(self.memory) > self.max_memory:
          del self.memory[0]

    # based on https://gist.github.com/EderSantana/c7222daa328f0e885093
    def exp_get_batch(self):
        len_memory = len(self.memory)
        n_examples = min(len_memory, self.batch_size)
        inputs = np.zeros((n_examples, self.n_input))
        targets = np.zeros((n_examples, self.n_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,size=n_examples)):
            #get_memory
            states = self.memory[idx]

            # input
            inputs[i] = states['state_t'].astype(np.float64)

            # targets - not correcting those which are not taken
            feed_dict = {self.x: states['state_t'].reshape(1,-1)}
            targets[i] = self.session.run(self.predictor, feed_dict)
            
            # acted action
            feed_dict = {self.x: states['state_tp1'].reshape(1,-1)}
            Qsa = np.max(self.session.run(self.predictor, feed_dict))

            # check if endgame and if not use Bellman's equation
            if states['endgame']:
                targets[i,states['action']] = states['reward']
            else:
                targets[i,states['action']] = states['reward'] + self.discount * Qsa
        return inputs, targets

    # construct network
    def network_forward(self):
        layer_1 = tf.nn.relu(tf.matmul(self.x, self.weights['h1'] + self.biases['b1']))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.weights['h2'] + self.biases['b2']))  
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']    
        return out_layer
    
    # Construct model
    def define_model(self):
        self.predictor = self.network_forward()

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.pow(self.predictor-self.y, 2))/(2*self.batch_size, name = 'mse_mean')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.train_op = optimizer.minimize(loss, global_step=self.global_step)

        # Initializing the session
        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init)

        # Initializing SummaryWriter
        tf.scalar_summary(loss.op.name, loss)
        self.summary_op = tf.merge_all_summaries()

    # process reward
    def process_reward(self,ts,reward,endgame):
        if ts == self.n_steps-1 and endgame == True:
            reward = 1.0*self.n_steps
        return reward


    # Train loop
    def train_model(self):
        # start open ai monitor
        if self.monitor_file:
            self.env.monitor.start(self.monitor_file,force=True)

        # Training cycle
        for epoch in range(self.n_episodes):

            # restart episode
            state_tp1 = self.env.reset()
            endgame = False
            avg_cost = 0.
            avg_qval = 0.
            states = {}

            for t in range(self.n_steps):
                self.env.render()
                state_t1 = np.array(state_tp1)
        
                # epsilon-greedy exploration
                if np.random.rand() <= self.eps(epoch):
                    action = self.env.action_space.sample()
                else:
                    feed_dict = {self.x: state_t1.reshape(1,-1)}
                    qvals = self.session.run(self.predictor, feed_dict)
                    avg_qval += np.max(qvals)
                    action = np.argmax(qvals)

                # take a next step
                state_tp1, reward, endgame, info = self.env.step(action)

                # process reward
                # reward = self.process_reward(t,reward,endgame)

                #store experience
                states['action'] = action
                states['reward'] = float(reward)
                states['endgame'] = endgame
                states['state_t'] = np.array(state_t1)
                states['state_tp1'] = np.array(state_tp1)
                self.exp_remember(states)

                # Training loop
                if self.consec_wins < self.stop_training:
                    # get experience replay
                    x_batch, y_batch = self.exp_get_batch()
                    _, c = self.session.run([self.train_op, self.loss], feed_dict={self.x: x_batch, self.y: y_batch})
                    # Compute average loss
                    avg_cost += c
                    # TF summary writer
                    self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, self.session.graph)
                    self.summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(self.summary_str, epoch)

                # Check if lost or not
                if endgame == True:
                    if t == self.n_steps-1:
                        self.consec_wins +=1
                        print("{:4d}: won!".format(epoch))
                    else:
                        self.consec_wins = 0
                        print("{:4d}: lost after {:3d}, cost {:8.4f}, qval {:8.4f}".
                                format(epoch,t+1,avg_cost/t,avg_qval/t))
                        break

        # close monitor session
        if self.monitor_file:
            self.env.monitor.close()

    # submit result for the leaderboard
    def submit_result(self,algo_id,api_key):
        gym.upload(self.monitor_file,
            algorithm_id=algo_id,
            api_key=api_key,
            ignore_open_monitors=False)