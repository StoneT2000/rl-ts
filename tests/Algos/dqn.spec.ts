import { DQN } from '../../src/RL/Algos/dqn/index';
import { CartPole } from '../../src/RL/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs';
import * as random from '../../src/RL/utils/random';
import { expect } from 'chai';
describe('Test DQN', () => {
  it('should run', async () => {
    random.seed(0);
    const policyNet = tf.sequential();
    policyNet.add(tf.layers.dense({ units: 12, inputShape: [4], activation: 'tanh' }));
    policyNet.add(tf.layers.dense({ units: 24, activation: 'tanh' }));
    policyNet.add(tf.layers.dense({ units: 64, activation: 'tanh' }));
    policyNet.add(tf.layers.dense({ units: 2, activation: 'linear' }));
    const targetNet = tf.sequential();
    targetNet.add(tf.layers.dense({ units: 12, inputShape: [4], activation: 'tanh' }));
    targetNet.add(tf.layers.dense({ units: 24, activation: 'tanh' }));
    targetNet.add(tf.layers.dense({ units: 64, activation: 'tanh' }));
    targetNet.add(tf.layers.dense({ units: 2, activation: 'linear' }));
    targetNet.setWeights(policyNet.getWeights());
    const makeEnv = () => {
      return new CartPole();
    };

    const dqn = new DQN(makeEnv, {
      replayBufferCapacity: 10000,
      policyNet,
      targetNet,
    });

    const evaluationRewards = [];
    const evaluateModel = () => {
      let state = dqn.env.reset();
      let reward = 0;
      while (true) {
        const action = dqn.act(state);
        const stepInfo = dqn.env.step(action);
        reward += stepInfo.reward;
        state = stepInfo.observation;
        if (stepInfo.done) break;
      }
      evaluationRewards.push(reward);
    };
    await dqn.train({ epochCallback: evaluateModel, totalEpisodes: 10 });
    expect(evaluationRewards.length).to.equal(10);
  }).slow(2000);
});
