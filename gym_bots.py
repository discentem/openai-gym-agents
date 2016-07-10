from gym_models import ExperienceQModel


if __name__ == "__main__":

    # CartPole-v0
    model = ExperienceQModel(
        env='CartPole-v0',\
        monitor_file = None,\
        max_memory=10000,\
        discount=.95,\
        n_episodes=400,\
        n_steps=200,\
        batch_size=100,\
        learning_rate = 1.e-2,\
        exploration = lambda x: 0.1,\
        stop_training = 10
    )
    model.train_model()

    # model.submit_result(
        # algo_id='alg_BjyWQTp1TCq9zSVyEtXTA',\
        # api_key='sk_7355XBGvTiqI1GxxakqSYw')