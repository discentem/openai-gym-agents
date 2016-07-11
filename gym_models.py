import gym
import re
import tensorflow as tf
import numpy as np
import shutil

class ExperienceQModel(object):
    def __init__(self, env, monitor_file, log_dir, max_memory=10000, discount=.9, n_episodes=100, 
                 n_steps=100, batch_size=100, learning_rate = 0.01, dropout = 1.0,
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
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = int(re.findall('\d+',str(self.env.action_space))[0]) # shameless hack to get a dim of actions
        
        # training parameters
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.n_steps = n_steps # must be equal to episode length
        self.batch_size = batch_size
        self.stop_training = stop_training # stop training after stop_training consecutive wins
        self.consec_wins = 0 # number of consecutive wins to stop training
        self.global_step = 0 # global step

        # Neural Network Parameters
        self.n_hidden_1 = self.n_states
        
        # Initialize tensor flow parameters
        self.x = tf.placeholder(tf.float32, [None, self.n_states],name='states')
        self.y = tf.placeholder(tf.float32, [None, self.n_actions],name='qvals')
        self.keep_prob = dropout
        self.dropout = tf.placeholder(tf.float32,name='dropout')

        
        # Tensorboard directory
        try:
            shutil.rmtree(log_dir) #clean
        except:
            pass
        self.log_dir = log_dir
        
        # define graph
        self.tf_define_model()

    # process reward
    def exp_process_reward(self,ts,reward,endgame):
        # if ts < self.n_steps-1 and endgame == True:
        #     reward = -1.0*ts/5.0 #penalize last moves
        # if ts == self.n_steps-1 and endgame == True:
        #     reward = ts #win
        # if endgame == True:
            # reward = 1.0*ts
        return reward

    # saving to memory
    def exp_save_to_memory(self, states):
        self.memory.append(states.copy())
        if len(self.memory) > self.max_memory:
          del self.memory[0]

    # based on https://gist.github.com/EderSantana/c7222daa328f0e885093
    def exp_get_batch(self):
        len_memory = len(self.memory)
        n_examples = min(len_memory, self.batch_size)
        inputs = np.zeros((n_examples, self.n_states))
        targets = np.zeros((n_examples, self.n_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,size=n_examples)):
            #get_memory
            states = self.memory[idx]

            # input
            inputs[i] = states['state_t'].astype(np.float32)

            # targets - not correcting those which are not taken
            feed_dict = {self.x: states['state_t'].reshape(1,-1), self.dropout: 1.0}
            targets[i] = self.session.run(self.predictor, feed_dict)
            
            # acted action
            feed_dict = {self.x: states['state_tp1'].reshape(1,-1), self.dropout: 1.0}
            Qsa = np.max(self.session.run(self.predictor, feed_dict))

            # check if endgame and if not use Bellman's equation
            if states['endgame']:
                targets[i,states['action']] = states['reward']
            else:
                targets[i,states['action']] = states['reward'] + self.discount * Qsa
        return inputs, targets
    
    # aux to define a weight variable
    def tf_weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    # aux to define a bias
    def tf_bias_variable(self,shape):
        initial = tf.constant(.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    # aux to attach many summaries
    def tf_variable_summaries(self,var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

    # Aux function to define layers
    def tf_nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('inputs'):
                self.tf_variable_summaries(input_tensor, layer_name + '/input')

            with tf.name_scope('weights'):
                weights = self.tf_weight_variable([input_dim, output_dim])
                self.tf_variable_summaries(weights, layer_name + '/weights')

            with tf.name_scope('biases'):
                biases = self.tf_bias_variable([output_dim])
                self.tf_variable_summaries(biases, layer_name + '/biases')

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.add(tf.matmul(input_tensor, weights),biases)
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                activations = act(preactivate, 'activation')
                tf.histogram_summary(layer_name + '/activations', activations)

            return activations


    # construct network
    def tf_network(self):
        hidden1 = self.tf_nn_layer(self.x, self.n_hidden_1, self.n_hidden_1, 'layer1', act=tf.nn.relu)

        with tf.name_scope('dropout'):
            tf.scalar_summary('dropout_probability', self.dropout)
            dropped = tf.nn.dropout(hidden1, self.dropout)

        out = self.tf_nn_layer(dropped, self.n_hidden_1, self.n_actions, 'out', act=tf.identity)
        return out
    

    # Construct model
    def tf_define_model(self):
        
        # Init session
        self.session = tf.Session()

        # Model scope
        with tf.name_scope('Model'):
            self.predictor = self.tf_network()

        # Loss
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_sum(tf.pow(self.predictor-self.y, 2)/(2*self.batch_size))

        # Define optimizer
        with tf.name_scope('SGD'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Prepare summaries
        tf.scalar_summary('loss', self.loss)

        # Summary writer
        self.merged_summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.log_dir + '/train', graph=tf.get_default_graph())

        # Initializing the session
        self.session.run(tf.initialize_all_variables())


    # Train loop
    def tf_train_model(self):
        # start open ai monitor
        if self.monitor_file:
            self.env.monitor.start(self.monitor_file,force=True)

        # Training cycle
        for epoch in range(self.n_episodes):

            # restart episode
            state_tp1 = self.env.reset()
            endgame = False
            avg_loss = 0.
            avg_max_qval = 0.
            states = {}

            for t in range(self.n_steps):
                self.env.render()
                state_t1 = np.array(state_tp1)
        
                # epsilon-greedy exploration
                if self.consec_wins < self.stop_training and np.random.rand() <= self.eps(epoch):
                    action = self.env.action_space.sample()
                else:
                    feed_dict = {self.x: state_t1.reshape(1,-1), self.dropout: 1.0}
                    qvals = self.session.run(self.predictor, feed_dict)
                    avg_max_qval += np.max(qvals)
                    action = np.argmax(qvals)

                # take a next step
                state_tp1, reward, endgame, info = self.env.step(action)

                # process reward
                reward = self.exp_process_reward(t,reward,endgame)

                #store experience
                states['action'] = action
                states['reward'] = float(reward)/10
                states['endgame'] = endgame
                states['state_t'] = np.array(state_t1)
                states['state_tp1'] = np.array(state_tp1)
                self.exp_save_to_memory(states)

                # Training loop
                if self.consec_wins < self.stop_training:
                    # get experience replay
                    x_batch, y_batch = self.exp_get_batch()
                    # create feed dictionary
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.dropout: self.keep_prob}
                    # training
                    _, loss, summary = self.session.run([self.train_op, self.loss, self.merged_summary_op],
                        feed_dict=feed_dict)
                    # add summary to the summary_writer
                    self.global_step += x_batch.shape[0]
                    self.summary_writer.add_summary(summary,self.global_step)
                    # avg loss
                    avg_loss += loss

                # Check if lost or not
                if endgame == True:
                    if (t == self.n_steps-1):
                        self.consec_wins +=1
                        print("{:4d}: won!".format(epoch+1))
                        break
                    else:
                        self.consec_wins = 0
                        print("{:4d}: lost after {:3d}, cost {:8.4f}, qval {:8.4f}".
                                format(epoch+1,t+1,avg_loss/t,avg_max_qval/t))
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