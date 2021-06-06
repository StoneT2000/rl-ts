import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs';
import * as random from '../../src/utils/random';
describe('Test VPG', () => {
  it('should run', async () => {
    random.seed(0);
    const makeEnv = () => {
      return new CartPole();
    };
    const env = makeEnv();
    const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48]);
    const vpg = new RL.Algos.VPG(makeEnv, ac, {
      actionToTensor: (action: tf.Tensor) => {
        return action.argMax(1);
      },
    });
    await vpg.train({
      steps_per_epoch: 100,
      epochs: 10,
      train_pi_iters: 10,
      train_v_iters: 80,
    });
  }).slow(2000);
});
