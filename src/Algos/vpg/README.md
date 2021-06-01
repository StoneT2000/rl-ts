# Vanilla Policy Gradient

This is an implementation of the vanilla policy gradient algorithm (VPG). It works with continuous observation spaces and both discrete and continuous action spaces.

For a sample script of using VPG on the CartPole environment, see https://github.com/StoneT2000/src/tree/main/examples/VPG/cartpole.js

Note that you need to create your own Actor Critic model which you can find settings for at https://github.com/StoneT2000/src/tree/main/src/Models/ac.ts

By default, the actor will produce continuous actions. The example shows you how to discretize the actions by providing a `actionToTensor` parameter.

To learn how VPG works, see https://spinningup.openai.com/en/latest/algorithms/vpg.html

## Customization

To customize VPG further and add features that aren't possible through configuration, read this page on [customization](https://github.com/StoneT2000/rl-ts/wiki/Customization)