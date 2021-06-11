# Deep Q Network

This is an implementation of Deep Q Networks. This will train a policy network that then selects discrete actions and optimizes expected return. The DQN algorithm here expects TensorFlow based models.

For a sample script of using DQN on the CartPole environment, see https://github.com/StoneT2000/rl-ts/blob/main/examples/DQN/cartpole.js

To initialize the model, you must pass a makeEnv function that creates the environment to train on. You must also provide your own policy network and target networks and they must be the same exact network architecture. You may also optionally pass in a replay buffer capacity, default capacity is `1000`

```js
const makeEnv = () => {
    return new RL.Environments.Examples.CartPole();
  };
const dqn = new RL.Algos.DQN(makeEnv, {
  replayBufferCapacity: 1000,
  policyNet,
  targetNet,
});
```
<!-- TODO - add documentation for the configurations -->
To train DQN, there are many configurations that can be provided into the `dqn.train(configs)` function that then trains the model on the given environment. See https://github.com/StoneT2000/rl-ts/blob/main/src/Algos/dqn/index.ts for options. Full documentation coming soon.

```js
dqn.train({ totalEpisodes: 1000, batchSize: 128, verbose: true, epochCallback, });
```

To then select an action using the current model, use
```js
dqn.act(obs)
```
where `obs` is the observation returned by the environment. The default act function will return the argmax of the policy network output on `obs`.

<!-- TODO - add CLI way to run this -->

## Customization

To customize DQN further and add features that aren't possible through configuration, read this page on [customization](https://github.com/StoneT2000/rl-ts/wiki/Customization)
