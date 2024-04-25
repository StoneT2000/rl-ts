import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs';
import * as random from '../../src/utils/random';
describe('Test PPO', () => {
  it('should run', async () => {
    random.seed(0);
    const makeEnv = () => {
      return new CartPole();
    };
    const env = makeEnv();
    const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48]);
    const ppo = new RL.Algos.PPO(makeEnv, ac, {
      actionToTensor: (action: tf.Tensor) => {
        return action.squeeze();
      },
    });
    await ppo.train({
      steps_per_iteration: 1024,
      iterations: 5,
    });
  }).slow(20000);
});
