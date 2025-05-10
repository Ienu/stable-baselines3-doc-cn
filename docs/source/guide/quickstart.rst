.. _quickstart:

===========
快速入门指南
===========

.. note::

  Stable-Baselines3 (SB3) 内部使用 :ref: `向量化环境(VecEnv) <vec_env>`。
  请阅读相关章节了解其特性以及与单个Gym环境的区别。


本库大部分接口遵循类似scikit-learn的强化学习算法语法。

以下是在CartPole环境中训练和运行A2C算法的快速示例：

.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import A2C

  env = gym.make("CartPole-v1", render_mode="rgb_array")

  model = A2C("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      vec_env.render("human")
      # 向量化环境会自动重置
      # if done:
      #   obs = vec_env.reset()

.. note::

	关于日志输出和命名的详细说明，请参考 :ref:`Logger <logger>` 章节.


`如果环境已在Gymnasium注册 <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs>`_ 且策略也已注册，只需一行代码即可训练模型：

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", "CartPole-v1").learn(10000)