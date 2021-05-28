import { DQN } from '../../src/RL/Algos/dqn/index';
import * as tf from '@tensorflow/tfjs';
import { ExtractActionType, ExtractObservationType, ExtractStateType } from '../../src/RL/Environments';
import * as np from '../../src/RL/utils/np';
import * as random from '../../src/RL/utils/random';
import { expect } from 'chai';
import nj from 'numjs';
import * as RL from '../../src';
describe('Test Pendulum', () => {
  it('should run', async () => {
    const env = new RL.Environments.Examples.Pendulum();
    env.maxEpisodeSteps = 10;
    let state = env.reset();
    while (true) {
      const action = env.actionSpace.sample();
      const { reward, observation, done, info } = env.step(action);
      state = observation;
      expect(state.shape).to.eql([3]);
      if (done) break;
    }
  });
});