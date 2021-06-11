# Proximal Policy Optimization

This is an implementation of the Proximal Policy Optimization algorithm (PPO). It works with continuous observation spaces and both discrete and continuous action spaces.

For a sample script of using PPO on the CartPole environment, see https://github.com/StoneT2000/rl-ts/blob/main/examples/ppo/cartpole.js

Note that you need to create your own Actor Critic model which you can find settings for at https://github.com/StoneT2000/rl-ts/blob/main/src/Models/ac.ts

By default, the actor will produce continuous actions. The example shows you how to discretize the actions by providing a `actionToTensor` parameter.

To learn how PPO works, see https://spinningup.openai.com/en/latest/algorithms/ppo.html

## Customization

To customize PPO further and add features that aren't possible through configuration, read this page on [customization](https://github.com/StoneT2000/rl-ts/wiki/Customization)