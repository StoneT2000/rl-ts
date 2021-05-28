# Typescript Reinforcement Learning Library 🤖

[![npm version](https://badge.fury.io/js/rl-ts.svg)](https://badge.fury.io/js/rl-ts)

This is a RL gym and library built with **typescript**, enabling faster bug-free development, powerful web visuals, and a gateway to developing and comparing reinforcement learning algorithms on the web and / or with node.js

![](https://github.com/StoneT2000/rl-ts/raw/main/src/RL/Environments/cartpole.gif)

Install with

```
npm install rl-ts
```

## Features

- Uses a **Open AI Gym** like interface to provide an accessible and general approach for developing environments and testing models

- Provides [standard integrated environments](https://github.com/StoneT2000/rl-ts/tree/main/src/RL/Environments) like CartPole

- Provides a toolbox of baseline agents such as DQN, as well as other algorithms like policy iteration.

- **typescript** means types, less bugs, and powerful, easily buildable, visuals.

Why RL with typescript / javascript (TS/JS)? TS/JS powers the web and a ton of the awesome visuals you see online.
While indeed production level reinforcement deep learning should be done with python / C++, TS/JS makes it easy to directly integrate RL
into the browser and serve not only as a simple integration for visualizers, but also as a powerful web-based teaching tool on RL. Moreover, this library can help developers with a background in TS/JS ease their way into using python / C++ for RL.

Inspired by [Andrej Karpathy's blog post](http://karpathy.github.io/2016/05/31/rl/), I'm building this to get a strong, end to end understanding of deep learning and reinforcement learning. While one could just use his library or some of the other ones out there, none of them are built with typescript / actively maintained, nor is there really an emphasis on structured environments and viewers that leverage TS/JS. Typescript enables typing which massively improves the scalability and maintainability of this library and enables a much more in depth TS/JS based RL library.

## Getting Started

This library contains both integrated environments and various algorithms.

To use environments, you can then create a new environment and step through it and render it. The following code produces the replay shown earlier in the readme, playing out 100 episodes of the CartPole environment and opening a web based viewer to render the environment.

```js
const RL = require('rl-ts');
const env = new RL.Environments.Examples.CartPole();

const main = async () => {
  for (let episode = 0; episode < 100; episode++) {
    let state = env.reset();
    while (true) {
      const action = env.actionSpace.sample();
      const { reward, observation, done, info } = env.step(action);
      state = observation;
      await env.render('web', { fps: 60, episode });
      if (done) break;
    }
  }
};
main();
```

To use algorithms, currently there are Algorithms and Dynammic Programming algorithms. At the core, the package relies on the [numjs](https://github.com/nicolaspanel/numjs) package (js numpy) and [tensorflow js (tfjs)](https://github.com/tensorflow/tfjs) for computations. By default, almost all data is stored in numjs arrays with data being stored in tfjs tensors only if necessary (e.g. for neural nets, autograd, optimization).

```js
const RL = require('rl-ts');
const makeEnv = () => new RL.Environments.Examples.CartPole();
// RL.Algos has Q-learning and policy gradient based methods
const dqn = new RL.Algos.DQN(makeEnv, configs); // create a dqn model to then train policy / target networks
// RL.DP has DP based methods
const policyIteration = new RL.DP.PolicyIteration(makeEnv, configs); // create a policyIteration object to then run training
```

At the moment, the following algorithms are implemented:

- Q-Learning
  - [DQN](https://github.com/StoneT2000/rl-ts/tree/main/src/RL/Algos/dqn)
- DP Methods
  - [Policy Evaluation](https://github.com/StoneT2000/rl-ts/tree/main/src/RL/DP)
  - [Policy Iteration](https://github.com/StoneT2000/rl-ts/tree/main/src/RL/DP)

## Road map and plans

This library aims to be effectively a combination of the Open AI Gym and a miniature Stable Baselines, with more emphasis on naive RL algorithms to be used as a educational resource and help build more powerful and accessible environment viewers.

## Development

First install all necessary dependencies via

```
npm i
```

To build the library run

```
npm run build
```

To run tests run

```
npm run tests
```
