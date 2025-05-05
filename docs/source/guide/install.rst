.. _install:

安装
============


前置条件
-------------

Stable-Baselines3 需要 python 3.9+ 和 PyTorch >= 2.3

Windows
~~~~~~~

建议 Windows 用户使用 `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_  以便更轻松地安装 Python 包和所需库。您需要准备 Python 3.8 或更高版本的环境。  

快速开始可直接跳转至下一步安装 Stable-Baselines3。

.. note::

	尝试创建 Atari 环境时可能会出现与缺失 DLL 文件和模块相关的模糊错误。这是 atari-py 包的问题。 `更多信息请参考 <https://github.com/openai/atari-py/issues/65>`_.


稳定版安装
~~~~~~~~~~~~~~
使用 pip 安装 Stable Baselines3：

.. code-block:: bash

    pip install stable-baselines3[extra]

.. note::
        某些 shell（如 Zsh）需要在方括号外加引号： ``pip install 'stable-baselines3[extra]'`` `详情参见 <https://stackoverflow.com/a/30539963>`_.


此命令会安装可选依赖项（如 Tensorboard、OpenCV 或用于 Atari 游戏的 ``ale-py``）。若不需要这些功能，可使用：

.. code-block:: bash

    pip install stable-baselines3


.. note::

  如果需要在无 X-server 的环境中使用 OpenCV（例如在 Docker 镜像内），需安装 ``opencv-python-headless``, 详见 `issue #298 <https://github.com/DLR-RM/stable-baselines3/issues/298>`_.


最新版本安装
---------------------

.. code-block:: bash

	pip install git+https://github.com/DLR-RM/stable-baselines3

安装含额外功能的版本：

.. code-block:: bash

  pip install "stable_baselines3[extra,tests,docs] @ git+https://github.com/DLR-RM/stable-baselines3"


开发版安装
-------------------

如需参与 Stable-Baselines3 开发（支持运行测试和构建文档）：

.. code-block:: bash

    git clone https://github.com/DLR-RM/stable-baselines3 && cd stable-baselines3
    pip install -e .[docs,tests,extra]


使用 Docker 镜像
-------------------

如需已预装 stable-baselines 的 Docker 镜像，
推荐使用 `RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ 提供的镜像。

以下镜像仅包含 stable-baselines3 的依赖项（不含本体），专为开发设计。

使用预构建镜像
~~~~~~~~~~~~~~~~

GPU 镜像 (需要 `nvidia-docker`_):

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3

仅 CPU 版本：

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3-cpu

构建 Docker 镜像
~~~~~~~~~~~~~~~~~~~~~~~~

构建 GPU 镜像 (使用 nvidia-docker):

.. code-block:: bash

   make docker-gpu

构建 CPU 镜像：

.. code-block:: bash

   make docker-cpu

注意：使用代理时需要传递额外参数并进行 `tweaks`_:

.. code-block:: bash

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

运行镜像 (CPU/GPU)
~~~~~~~~~~~~~~~~~~~~~~~~

运行 nvidia-docker GPU 镜像

.. code-block:: bash

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3 bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

或使用 shell 脚本：

.. code-block:: bash

   ./scripts/run_docker_gpu.sh pytest tests/

运行 docker CPU 镜像：

.. code-block:: bash

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3-cpu bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

或使用 shell 脚本：

.. code-block:: bash

   ./scripts/run_docker_cpu.sh pytest tests/

Docker 命令解析：

-  ``docker run -it`` 创建镜像实例（容器）并以交互模式运行（支持 ctrl+c 中断） 
-  ``--rm`` 表示退出/停止后自动删除容器（否则需手动执行 ``docker rm``）
-  ``--network host`` 不使用网络隔离，允许在宿主机使用 tensorboard/visdom
-  ``--ipc=host`` 使用宿主机的 IPC (POSIX/SysV IPC) 命名空间，实现共享内存段、信号量和消息队列的隔离
-  ``--name test`` 显式命名容器为 ``test``（否则将随机分配名称） 
-  ``--mount src=...`` 将本地目录（ ``pwd`` 命令结果）映射到容器内（路径为 ``/home/mamba/stable-baselines3``），容器内该路径下生成的日志文件将被保留  
-  ``bash -c '...'`` 在容器内执行命令，此处为运行测试（ ``pytest tests/`` ）

.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _tweaks: https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy