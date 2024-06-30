# GameAlpha

该项目是中山大学人工智能学院博弈论课程期末大作业——对Alphaholdem的模型复现。我们选择了pytorch深度学习框架和ray强化学习框架来进行复现。除了参考Alphaholdem论文中的细节，我们还参考了一个tensorflow版本的复现[An unoffical implementation of *AlphaHoldem*](https://github.com/bupticybee/AlphaNLHoldemhttps://)。在此基础上，我们主要做了如下更新：

1. 实现了pytorch版本的模型复现
2. 重构了深度学习网络，应用了残差神经网络
3. 将ray更新至2.7.2，并使代码能在新版本上运行。
4. 更新了损失函数以及模型中的策略，使之更好地符合论文中的公式

# Train

配置好环境后，运行下列代码即可开始训练：

```
python train_league.py
```

若想修改参数，有下列两种方法：

1. 修改confs文件夹中的nl_holdem.py文件
2. 在命令中增加参数，具体细节可以查看train_league.py文件

# 个人留言

大家好，我是主要负责模型复现的同学。由于时间、计算资源、个人能力的限制，该模型复现还有很大的进步空间。因此未来我还会继续更新该项目，将要做的工作可能有：

1. 完善ray框架。由于对ray的不熟悉，加上ray更新频繁，使得ray的编程范式出现了很大变化，之后修改代码使代码更好地适配新版本。
2. 完善网络。尝试不同的网络结构。
3. 模型对抗。尝试将模型与其他模型进行对抗。
