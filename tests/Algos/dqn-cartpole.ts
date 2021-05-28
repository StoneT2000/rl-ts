import {DQN} from "../../src/RL/Algos/dqn/index";
import {CartPole} from "../../src/RL/Environments/examples/cartpole";
import * as tf from '@tensorflow/tfjs';
import { ExtractActionType, ExtractStateType } from "../../src/RL/Environments";
import * as np from "../../src/RL/utils/np";
import nj from 'numjs';
import * as random from "../../src/RL/utils/random";
import { env, Scalar } from "@tensorflow/tfjs";
random.seed(0);
const policyNet = tf.sequential();
policyNet.add(tf.layers.dense({units: 12, inputShape: [4], activation: 'tanh'}));
policyNet.add(tf.layers.dense({units: 24, activation: 'tanh'}));
policyNet.add(tf.layers.dense({units: 64, activation: 'tanh'}));
policyNet.add(tf.layers.dense({units: 2, activation: 'linear' }))
const targetNet = tf.sequential();
targetNet.add(tf.layers.dense({units: 12, inputShape: [4], activation: 'tanh'}));
targetNet.add(tf.layers.dense({units: 24, activation: 'tanh'}));
targetNet.add(tf.layers.dense({units: 64, activation: 'tanh'}));
targetNet.add(tf.layers.dense({units: 2, activation: 'linear' }))
targetNet.setWeights(policyNet.getWeights());
const makeEnv = () => {
  return new CartPole();
}

const dqn = new DQN(makeEnv, {
  replayBufferCapacity: 10000,
  policyNet,
  targetNet,
  stateToTensor: (state: ExtractStateType<CartPole>) => {
    return np.toTensor(state).reshape([1, 4]);
  },
  actionToTensor: (action: ExtractActionType<CartPole>) => {
    action
    return tf.tensor(action);
  },
});

const evaluateModel = (data: {time: number, episodeRewards: number[]}) => {
  let state = dqn.env.reset();
  let reward = 0;
  while(true) {
    let action = dqn.act(state);
    const stepInfo = dqn.env.step(action);
    reward += stepInfo.reward;
    state = stepInfo.observation;
    if (stepInfo.done) break;
  }
  console.log(`Episode ${data.episodeRewards.length} - Step ${data.time} - train ${data.episodeRewards[data.episodeRewards.length - 1]} - eval: Rewards ${reward}`);
};
dqn.train({ epochCallback: evaluateModel });
