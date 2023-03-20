const tf = require('@tensorflow/tfjs-node');
const RL = require('../../lib');

const RUN = `cart-2-categorical`;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

const main = async () => {
  // seed for reproducibility
  RL.random.seed(0);

  const steps_per_iteration = 2048;

  // create a makeEnv function. It should return a new environment object that the VPG algorithm will interact with
  const makeEnv = () => {
    return new RL.Environments.Examples.CartPole();
  };
  const env = makeEnv();

  // create the Actor Critic model
  const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48]);

  // create the ppo algorithm and define a actionToTensor function to discretize the actions produced by the Actor
  const ppo = new RL.Algos.PPO(makeEnv, ac, {
    actionToTensor: (action) => {
      // Cartpole has a discrete action space
      // Actor Critic uses categorical distribution for discrete action space to sample actions
      return action.squeeze();
    },
  });

  // define a evaluation function to be called at the end of every epoch
  const iterationCallback = async ({ iteration, ep_rets, kl, loss_pi, loss_vf }) => {
    // let obs = env.reset();
    // let rewards = 0;
    // while (true) {
    //   const action = ppo.act(obs);
    //   const stepInfo = env.step(action);
    //   rewards += stepInfo.reward;
    //   if (epoch > 10) {
    //     // after 10 epochs, start rendering the evaluation onto a web viewer
    //     await env.render('web', { fps: 60, episode: epoch });
    //   }
    //   obs = stepInfo.observation;
    //   if (stepInfo.done) break;
    // }
    // console.log(`Iteration ${iteration} - Eval Rewards: ${rewards}`);
    summaryWriter.scalar('kl', kl, iteration * steps_per_iteration);
    summaryWriter.scalar('reward', ep_rets.mean, iteration * steps_per_iteration);
    summaryWriter.scalar('loss_pi', loss_pi, iteration * steps_per_iteration);
    summaryWriter.scalar('loss_vf', loss_vf, iteration * steps_per_iteration);
  };

  // Uncomment the 2 lines below to train on 2 CPUs. Will train on forkCount + 1 cpus.
  // let forkCount = 1;
  // await RL.ct.fork(forkCount);

  // train the actor critic model with ppo
  ppo.train({
    verbose: true,
    steps_per_iteration,
    batch_size: 64,
    iterations: 100,
    n_epochs: 10,
    lam: 0.95,
    vf_coef: 0.5,
    iterationCallback,
  });
};
main();
