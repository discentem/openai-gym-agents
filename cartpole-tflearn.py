from __future__ import print_function
import gym
import tflearn
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
        self.n_actions = self.env.action_space.n
        
        # training parameters
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.n_steps = n_steps # must be equal to episode length
        self.batch_size = batch_size
        self.stop_training = stop_training # stop training after stop_training consecutive wins
        self.consecutive_wins = 0

        # Neural Network Parameters
        self.n_hidden_1 = self.n_states
        
        # Initialize tensor flow parameters
        self.x = tf.placeholder(tf.float32, [None, self.n_states],name='states')
        self.y = tf.placeholder(tf.float32, [None, self.n_actions],name='qvals')
        
        # Tensorboard directory
        try:
            shutil.rmtree(log_dir) #clean
        except:
            pass
        self.log_dir = log_dir
        
        # define graph
        self.tf_build_graph()
        self.tf_build_summaries()

    # process reward
    def exp_process_reward(self,ts,reward,endgame):
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
    def tf_network(self):
        net = tflearn.fully_connected(self.x, self.n_hidden_1, activation='relu',name='hidden1')
        qvalues = tflearn.fully_connected(net, self.n_actions,name='out')
        return qvalues

    # Construct graph
    def tf_build_graph(self):
        
        # Init session
        self.session = tf.Session()

        # Model scope
        self.predictor = self.tf_network()
        self.loss = tf.reduce_sum(tf.pow(self.predictor-self.y, 2)/(2*self.batch_size))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    # Set up some episode summary ops to visualize on tensorboard.
    # source - tflearn example
    def tf_build_summaries(self):
        episode_score = tf.Variable(0.)
        tf.scalar_summary("Episode Score", episode_score)
    
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary("Episode Avg Loss", episode_avg_loss)

        episode_avg_qmax = tf.Variable(0.)
        tf.scalar_summary("Episode Avg Qmax", episode_avg_qmax)
    
        episode_eps = tf.Variable(0.)
        tf.scalar_summary("Episode Epsilon", episode_eps)
    
        # Threads shouldn't modify the main graph, so we use placeholders
        # to assign the value of every summary (instead of using assign method
        # in every thread, that would keep creating new ops in the graph)
        summary_vars = [episode_score, episode_avg_loss, episode_avg_qmax, episode_eps]
        self.summary_placeholders = [tf.placeholder("float")
            for i in range(len(summary_vars))]
    
        self.assign_ops = [summary_vars[i].assign(self.summary_placeholders[i])
            for i in range(len(summary_vars))]
        
        self.summary_op = tf.merge_all_summaries()

    # Train loop
    def tf_train_model(self):
        # Initializing the session
        self.summary_writer = tf.train.SummaryWriter(self.log_dir + '/train', graph=tf.get_default_graph())
        self.session.run(tf.initialize_all_variables())

        # start open ai monitor
        if self.monitor_file:
            self.env.monitor.start(self.monitor_file,force=True)

        # Training cycle
        for epoch in range(self.n_episodes):

            # restart episode
            state_tp1 = self.env.reset()
            endgame = False
            states = {}

            # episode stats
            episode_stats = {}
            episode_stats['score'] = 0
            episode_stats['avg_max_q'] = 0
            episode_stats['avg_loss'] = 0
            episode_stats['epsilon'] = 0

            for t in range(self.n_steps):
                self.env.render()
                state_t1 = np.array(state_tp1)
        
                # epsilon-greedy exploration
                if self.consecutive_wins < self.stop_training and np.random.rand() <= self.eps(epoch):
                    action = self.env.action_space.sample()
                else:
                    feed_dict = {self.x: state_t1.reshape(1,-1)}
                    qvals = self.session.run(self.predictor, feed_dict)
                    episode_stats['avg_max_q'] += np.max(qvals)
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
                if self.consecutive_wins < self.stop_training:
                    # get experience replay
                    x_batch, y_batch = self.exp_get_batch()
                    # create feed dictionary
                    feed_dict = {self.x: x_batch, self.y: y_batch}
                    # training
                    _, loss = self.session.run([self.train_op, self.loss],
                        feed_dict=feed_dict)
                    
                    # avg loss
                    episode_stats['avg_loss'] += loss


                # Check if lost or not
                if endgame == True:
                    print ("{:4d}: score {:3d}, cost {:8.4f}, qval {:8.4f}, eps {:8.4f}".
                            format(epoch+1,\
                                episode_stats['score'],
                                episode_stats['avg_loss']/t,
                                episode_stats['avg_max_q']/t,
                                episode_stats['epsilon']))
                    for i in episode_stats:
                        self.session.run(self.assign_ops[i],\
                            {self.summary_placeholders[i]: float(episode_stats[i])})

                    break
                else:
                    episode_stats['score'] += 1

        # close monitor session
        if self.monitor_file:
            self.env.monitor.close()

    # submit result for the leaderboard
    def submit_result(self,algo_id,api_key):
        gym.upload(self.monitor_file,
            algorithm_id=algo_id,
            api_key=api_key,
            ignore_open_monitors=False)


if __name__ == "__main__":

    model = ExperienceQModel(
        env='CartPole-v0',\
        monitor_file = None,\
        log_dir = '/tmp/tf/cartpole-256_1e-3_norm',\
        max_memory=40000,\
        discount=.90,\
        n_episodes=400,\
        n_steps=200,\
        batch_size=10,\
        learning_rate = 1.e-3,\
        dropout = 1.0,\
        exploration = lambda x: 0.1 if x<50 else 0,\
        stop_training = 10
    )

    model.tf_train_model()

    # model.submit_result(
        # algo_id='',\
        # api_key='')