import { expect } from 'chai';
import * as RL from '../../src';
describe('Test Pendulum', () => {
  it('should run', async () => {
    const env = new RL.Environments.Examples.Pendulum();
    env.maxEpisodeSteps = 10;
    let state = await env.reset();
    while (true) {
      const action = env.actionSpace.sample();
      const { observation, done } = await env.step(action);
      state = observation;
      expect(state.shape).to.eql([3]);
      if (done) break;
    }
  });
});
