from gym_models import ExperienceQModel


if __name__ == "__main__":

    # model = ExperienceQModel(
    #     env='CartPole-v0',\
    #     monitor_file = 'results/cartpole',\
    #     log_dir = '/tmp/tf/cartpole-256_1e-3_norm',\
    #     max_memory=40000,\
    #     discount=.90,\
    #     n_episodes=400,\
    #     n_steps=200,\
    #     batch_size=256,\
    #     learning_rate = 1.e-3,\
    #     dropout = 1.0,\
    #     exploration = lambda x: 0.1 if x<50 else 0,\
    #     stop_training = 10
    # )

    model = ExperienceQModel(
        env='MountainCar-v0',\
        monitor_file = 'results/mountaincar',\
        log_dir = '/tmp/tf/mountaincar-256_1e-3_norm',\
        max_memory=40000,\
        discount=.90,\
        n_episodes=400,\
        n_steps=200,\
        batch_size=256,\
        learning_rate = 1.e-3,\
        dropout = 1.0,\
        exploration = lambda x: 0.1 if x<50 else 0,\
        stop_training = 10
    )
    model.tf_train_model()

    # model.submit_result(
        # algo_id='alg_BjyWQTp1TCq9zSVyEtXTA',\
        # api_key='sk_7355XBGvTiqI1GxxakqSYw')