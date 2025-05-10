.. _rl_tips:

====================
强化学习技巧与实践指南
====================

本节旨在帮助您开展强化学习实验。
内容涵盖RL通用建议（入门路径、算法选择、评估方法等），以及自定义环境使用技巧或RL算法的实现。

.. note::

  我们有 `YouTube视频 <https://www.youtube.com/watch?v=Ikngt0_DXJg>`_ 包括本章的更多细节。你也可以找到 `幻灯片 <https://araffin.github.io/slides/rlvs-tips-tricks/>`_.


.. note::

	我们也有 `设计和运行真实世界强化学习实验视频 <https://youtu.be/eZ6ZEpCi6D8>`_, 幻灯片 `能够被在网上找到 <https://araffin.github.io/slides/design-real-rl-experiments/>`_.


强化学习通用指南
===============

速览要点
--------

1. 系统学习RL和Stable Baselines3
2. 进行定量实验与超参数调优
3. 使用独立测试环境评估性能（注意环境包装器）
4. 提升训练预算以获得更好效果


与其他学科相同，使用RL前需先建立理论基础 (推荐入门资源 `资源页面 <rl.html>`_ )
我们也建议你阅读 Stable Baselines3 (SB3) 文档并完成 `教程 <https://github.com/araffin/rl-tutorial-jnrr19>`_。
该教程涵盖从基础用法到高级特性（如回调函数和环境包装器）。

强化学习的特殊性在于： 训练数据通过智能体与环境交互产生 (不同于监督学习的固定数据集).
这个依赖可能导致恶性循环： 如果智能体收集了低质量的数据 (例如，没有奖励的轨迹), 那么它不会改进并持续累积劣质轨迹。

这个因素解释了强化学习每次运行的结果可能不同 (即，随机种子发生变化).
鉴于此，你应该多跑几次来获得定量的结果。

强化学习中的良好结果通常取决于找到合适的超参数。最近的算法（PPO，SAC，TD3，DroQ）通常需要很少的超参数调整，但是， *不要指望默认的参数*在任何环境下都能工作。

因此，我们*强烈建议*您参观 `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ (或原始论文) 来调整超参数。
将强化学习应用于新问题的最佳实践是进行自动超参数优化。同样，这也包括在 `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

当将强化学习应用于自定义问题时，您应该始终对智能体的输入进行归一化 (例如，对于 PPO/A2C 使用 ``VecNormalize`` )
并查看在其他环境中完成的常见预处理 (例如对于 `Atari <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`_, frame-stack, ...).
有关自定义环境的更多建议，请参阅下面的 *创建自定义环境时的提示和技巧* 一段。


当前强化学习的局限性
-------------------

你必须意识到强化学习当前的 `局限性 <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_ 。


无模型强化学习算法 (即 SB 中实现的所有算法) 通常是 *采样不充分的*. 它们需要大量的样本 (有时是数百万次交互) 来学习有用的东西。
这就是为什么强化学习的大部分成功都是在游戏或仿真中取得的。例如，在 ETH Zurich 的这项 `工作 <https://www.youtube.com/watch?v=aTDkYFZFWug>`_，ANYmal 机器人仅在仿真中进行了训练，然后在现实世界进行测试。

作为一般建议，为了获得更好的表现，您应该增加智能体的开销（训练步数）。

为了实现预期的行为，通常需要专家知识来设计足够的奖励函数。
此 *reward 工程* (或 `Freek Stulp <http://www.freekstulp.net/>`_ 创造的 *RewArt* ), 需要多次迭代。作为奖励塑造的一个很好的例子，你可以看看 `Deep Mimic 论文 <https://xbpeng.github.io/projects/DeepMimic/index.html>`_ 它结合了模仿学习和强化学习来做杂技动作。

强化学习的最后一个局限性是训练的不稳定性。也就是说，你可以在训练中观察到表现的大幅下降。
这种行为在 ``DDPG`` 中尤为明显, 这就是为什么它的扩展 ``TD3`` 试图解决这个问题。
其他方法，如 ``TRPO`` 或 ``PPO`` 利用 *trust region* 通过避免过大的更新来缩减该问题。


如何评估强化学习算法？
--------------------

.. note::

  在评估您的智能体并将结果与其他智能体的结果进行比较时，请注意环境包装。对 episode rewards 的修改或者长度也可能影响评估导致不理想的结果。参考 :ref:`Evaluation helper<eval>` 章节的 ``evaluate_policy`` 辅助函数。

由于大多数算法在训练过程中使用探索噪声，因此您需要一个单独的测试环境来评估智能体在给定时间的性能。建议定期评估您的代智能体的 ``n`` 个测试幕（ ``n``通常在5到20之间），并平均每一幕的奖励，以便做出准确的估计。

.. note::

	我们提供了一个 ``EvalCallback`` 来进行这样的评估。更多内容在 :ref:`Callbacks <callbacks>` 章节。

由于默认情况下某些策略是随机的 (例如 A2C 或 PPO), 你也应该尝试设置 `deterministic=True` 当调用 `.predict()` 方法，这经常能够得到更好的性能。
观察训练曲线 (时间步的幕奖励函数) 是一个很好的指标但低估了智能体的真实性能。


我们强烈推荐阅读 `Empirical Design in Reinforcement Learning <https://arxiv.org/abs/2304.01315>`_ ，因为它为运行强化学习实验时的最佳实践提供了有价值的见解。

我们也建议阅读 `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_ 其对于强化学习评估的一个很好的讨论，以及 `Rliable: Better Evaluation for Reinforcement Learning <https://araffin.github.io/post/rliable/>`_ 对于结果对比。

你也可以看看 Cédric Colas 的 `blog post <https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457>`_
以及 `issue <https://github.com/hill-a/stable-baselines/issues/199>`_ 。


我应该使用哪种算法？
==================

强化学习中没有灵丹妙药，你可以根据自己的需求和问题选择一个或另一个。第一个区别来自你的动作空间，即你有离散的（例如左、右……）吗？还是连续的动作（例如：达到一定的速度）？

一些算法仅针对一个或另一个领域量身定制： ``DQN`` 仅支持离散动作，而 ``SAC`` 仅限于连续动作。

第二个区别将帮助你决定是否可以并行化你的训练。
如果重要的是挂钟训练时间，那么你应该倾向于 ``A2C`` 及其衍生物（PPO，……）。
查看 `Vectorized Environments <vec_envs.html>`_，了解更多关于多头训练信息。

为了加速训练，你还可以看看 `SBX`_，它是SB3+Jax，它的功能比SB3少，但由于梯度更新的JIT编译，它可以比SB3-PyTorch快20倍。

在稀疏奖励设置中，我们建议使用HER（见下文）等专用方法或ARS（可在我们的 :ref:`contrib repo <sb3_contrib>`中找到）等基于群体的算法。

总结一下：

离散动作
--------

.. note::

	这涵盖了 ``Discrete``, ``MultiDiscrete``, ``Binary`` 和 ``MultiBinary`` 空间


离散动作-单个过程
^^^^^^^^^^^^^^^^

推荐具有扩展的 ``DQN`` (double DQN, 优先回放，...) 算法。
我们特别提供了 ``QR-DQN`` 在我们的 :ref:`contrib repo <sb3_contrib>`.
``DQN`` 通常训练得较慢 (根据挂钟时间) 但它是最有效的采样 (因为它的回放缓冲区).

离散动作-多个过程
^^^^^^^^^^^^^^^^

您应该尝试 ``PPO`` 或 ``A2C``.


连续动作
--------

连续动作-单个过程
^^^^^^^^^^^^^^^^

当前最先进的 (SOTA) 算法是 ``SAC``, ``TD3``, ``CrossQ`` 和 ``TQC`` (在我们的 :ref:`contrib repo <sb3_contrib>` 和 :ref:`SBX (SB3 + Jax) repo <sbx>`)。
请使用超参数 `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ 以获得最佳结果。

如果您想要一个非常有效的样本算法，我们建议使用 `DroQ configuration <https://twitter.com/araffin2/status/1575439865222660098>`_ 在 `SBX`_ 中(它在环境中每一步执行许多梯度步骤).


连续动作-多个过程
^^^^^^^^^^^^^^^^

查看 ``PPO``, ``TRPO`` (在我们的 :ref:`contrib repo <sb3_contrib>` 中) 或 ``A2C``。同样，别忘了选取超参数 `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ 对于连续动作问题 (cf *Bullet* envs).

.. note::

  归一化对于这些算法至关重要



目标环境
--------

如果您的环境遵循 ``GoalEnv`` 接口 (cf :ref:`HER <her>`), 则应使用 HER + (SAC/TD3/DDPG/DQN/QR-DQN/TQC) 根据动作空间。


.. note::

	``batch_size`` 是用于实验的重要超参数 :ref:`HER <her>`



创建自定义环境时的提示和技巧
==========================

如果您想了解如何创建自定义环境，我们建议您阅读以下内容 `page <custom_env.html>`_.
我们也提供了一个 `colab notebook <https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb>`_ 有关创建自定义 gym 环境的具体示例。

一些基本建议：

- 如果可以，即如果知道边界，请始终规范化观测空间
- 规范化您的动作空间，如果它是连续的，则使其对称（请参阅下面的潜在问题）。一个好的实践是重新缩放您的动作，使其位于[-1，1]。这不会限制您，因为您可以轻松地在环境中重新缩放操作
- 从成型的奖励 (即信息奖励) 和问题的简化版本开始
- 使用随机动作进行调试，以检查您的环境是否工作并遵循 gym 的接口（使用 ``check_env``，请参见下文）

创建定制环境时要记住两件重要的事情，即避免破坏马尔可夫假设，并正确处理由于超时（一幕中的最大步骤数）而导致的终止。
例如，如果动作和观测之间存在时间延迟（例如，由于wifi通信），则应提供观测历史记录作为输入。

因超时而终止（每幕最大步数）需要单独处理。
您应该返回 ``truncated = True``.
如果您使用 gym 的 ``TimeLimit`` 包装，这将自动完成。
你可以阅读 `Time Limit in RL <https://arxiv.org/abs/1712.00378>`_， 看一下 `Designing and Running Real-World RL Experiments video <https://youtu.be/eZ6ZEpCi6D8>`_ 或 `RL Tips and Tricks video <https://www.youtube.com/watch?v=Ikngt0_DXJg>`_ 来了解更多细节。


我们提供了一个帮助程序来检查您的环境是否正常运行：

.. code-block:: python

	from stable_baselines3.common.env_checker import check_env

	env = CustomEnv(arg1, ...)
	# 它将检查您的自定义环境，并在需要时输出其他警告
	check_env(env)


如果要在环境中快速尝试随机智能体，还可以执行以下操作：

.. code-block:: python

  env = YourEnv()
  obs, info = env.reset()
  n_steps = 10
  for _ in range(n_steps):
      # 随机动作
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      if done:
          obs, info = env.reset()


**为什么要规范化动作空间？**


大多数强化学习算法依赖于 `高斯分布 <https://araffin.github.io/post/sac-massive-sim/>`_ (以0为中心初始化，1为标准差)对于连续动作。
因此，如果在使用自定义环境时忘记规范化动作空间，这可能 `损害学习 <https://araffin.github.io/post/sac-massive-sim/>`_ 并且可能很难调试 (cf 附加图像和 `issue #473 <https://github.com/hill-a/stable-baselines/issues/473>`_)。

.. figure:: ../_static/img/mistake.png


使用高斯分布的另一个结果是动作范围不受限制。
这就是为什么clipping通常被用作绷带，以保持在有效的间隔。
更好的解决方案是使用挤压函数（cf ``SAC``）或Beta分布 (cf `issue #112 <https://github.com/hill-a/stable-baselines/issues/112>`_)。

.. note::

	这种说法对于 ``DDPG`` 或 ``TD3`` 不正确，因为它们不依赖于任何概率分布。



实现强化学习算法时的提示和技巧
============================

.. note::

  我们有 `YouTube上关于可靠强化学习的视频 <https://www.youtube.com/watch?v=7-PUg9EAa3Y>`_ 将更详细地介绍这一部分。 您还可以查找 `在线幻灯片 <https://araffin.github.io/slides/tips-reliable-rl/>`_.


当你试图实现算法来重现一篇强化学习论文时，John Schulman的 `强化学习研究的基本原则 <http://joschu.net/docs/nuts-and-bolts.pdf>`_ 是非常有用的 (`视频 <https://www.youtube.com/watch?v=8EcdaCk9KaQ>`_)。

我们 *建议遵循这些步骤以获得有效的强化学习算法*：

1. 多次阅读原文
2. 阅读现有实现 (如果有)
3. 尝试一些 "sign of life" 在玩具问题上
4. 通过使其在越来越困难的环境中运行来验证实现（您可以将结果与RL zoo进行比较）。
   通常需要为该步骤运行超参数优化。

您需要特别注意正在操作的不同对象的shape (一个广播错误将安静地失败 cf. `issue #75 <https://github.com/hill-a/stable-baselines/pull/76>`_)以及何时停止梯度传播。

Don't forget to handle termination due to timeout separately (see remark in the custom environment section above),
you can also take a look at `Issue #284 <https://github.com/DLR-RM/stable-baselines3/issues/284>`_ and `Issue #633 <https://github.com/DLR-RM/stable-baselines3/issues/633>`_.

对于强化学习逐渐困难且连续动作的环境的一种选择（@arafin）：

1. Pendulum (易于求解)
2. HalfCheetahBullet (中等难度，局部极小，塑形的奖励)
3. BipedalWalkerHardcore (如果它在那一个上有效，那么您可以有一个cookie)

离散动作的强化学习：

1. CartPole-v1 (易于优于随机智能体，难以实现最大性能)
2. LunarLander
3. Pong (最简单的 Atari 游戏之一)
4. 其他 Atari games (例如 Breakout)

.. _SBX: https://github.com/araffin/sbx