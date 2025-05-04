.. Stable Baselines3 documentation master file, created by
   sphinx-quickstart on Thu Sep 26 11:06:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stable-Baselines3 文档 - 可靠的强化学习实现
========================================================================

`Stable Baselines3 (SB3) <https://github.com/DLR-RM/stable-baselines3>`_ 是一套基于 PyTorch 的强化学习算法可靠实现。
它是 `Stable Baselines <https://github.com/hill-a/stable-baselines>`_ 的下一代主要版本。


Github 仓库: https://github.com/DLR-RM/stable-baselines3

论文: https://jmlr.org/papers/volume22/20-1364/20-1364.pdf

RL Baselines3 Zoo (SB3 的训练框架): https://github.com/DLR-RM/rl-baselines3-zoo

RL Baselines3 Zoo 提供了一系列预训练智能体、训练脚本、评估工具、超参数调优、结果可视化及视频录制功能。

SB3 Contrib (实验性强化学习代码及最新算法): https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

SBX (SB3 + Jax): https://github.com/araffin/sbx


主要特性
--------------

- 所有算法采用统一架构
- 符合 PEP8 规范（代码风格一致）
- 详尽的函数与类文档
- 高覆盖率的测试与类型提示
- 代码整洁
- 支持 Tensorboard
- **每种算法的性能均经过验证​​** (参见各算法页面的 *结果* 部分)


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/rl_tips
   guide/rl
   guide/algos
   guide/examples
   guide/vec_envs
   guide/custom_policy
   guide/custom_env
   guide/callbacks
   guide/tensorboard
   guide/integrations
   guide/rl_zoo
   guide/sb3_contrib
   guide/sbx
   guide/imitation
   guide/migration
   guide/checking_nan
   guide/developer
   guide/save_format
   guide/export


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/a2c
  modules/ddpg
  modules/dqn
  modules/her
  modules/ppo
  modules/sac
  modules/td3

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/env_util
  common/envs
  common/distributions
  common/evaluation
  common/env_checker
  common/monitor
  common/logger
  common/noise
  common/utils

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog
  misc/projects


引用 Stable Baselines3
------------------------
如需在出版物中引用本项目:

.. code-block:: bibtex

  @article{stable-baselines3,
    author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
    title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
    journal = {Journal of Machine Learning Research},
    year    = {2021},
    volume  = {22},
    number  = {268},
    pages   = {1-8},
    url     = {http://jmlr.org/papers/v22/20-1364.html}
  }

注：如需引用 SB3 的特定版本，可使用 `Zenodo DOI <https://doi.org/10.5281/zenodo.8123988>`_.

参与贡献
------------

欢迎对改进强化学习基线感兴趣的开发者参与，目前仍有部分优化需求待完成。

可查阅仓库中的议题 `repository <https://github.com/DLR-RM/stable-baselines3/labels/help%20wanted>`_.

贡献前请先阅读 `CONTRIBUTING.md <https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md>`_.

索引与目录
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
