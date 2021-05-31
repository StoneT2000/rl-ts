const tf = require('@tensorflow/tfjs');
const RL = require('rl-ts');
const main = async () => {
  // seed for reproducibility
  RL.random.seed(0);

  // create both the policy and target networks. 
  const policyNet = tf.sequential();
  policyNet.add(tf.layers.dense({ units: 12, inputShape: [4], activation: 'tanh' }));
  policyNet.add(tf.layers.dense({ units: 24, activation: 'tanh' }));
  policyNet.add(tf.layers.dense({ units: 2, activation: 'linear' }));
  const targetNet = tf.sequential();
  targetNet.add(tf.layers.dense({ units: 12, inputShape: [4], activation: 'tanh' }));
  targetNet.add(tf.layers.dense({ units: 24, activation: 'tanh' }));
  targetNet.add(tf.layers.dense({ units: 2, activation: 'linear' }));
  targetNet.setWeights(policyNet.getWeights());

  // create a makeEnv function. It should return a new environment object that the DQN model will interact with
  const makeEnv = () => {
    return new RL.Environments.Examples.CartPole();
  };

  /**
   * Create the dqn object / model. This will hold the policy and target networks, as well as let you use it to select actions given observations and train
   */
  const dqn = new RL.Algos.DQN(makeEnv, {
    replayBufferCapacity: 10000,
    policyNet,
    targetNet,
  });

  /**
   * this evaluation script will be passed into the training function to then be called at the end of each episode
   * 
   * The function has the following arguments. It may be async if you like, DQN will automatically await this function.
   * epochCallback(epochDate: {
   *    step: number;
   *    episodeDurations: number[];
   *    episodeRewards: number[];
   *    episodeIteration: number;
   * }): any;
   */
  const epochCallback = async ({ episodeIteration, episodeRewards }) => {
    let state = dqn.env.reset();
    let rewards = 0;
    // keep stepping through the environment until it is done;
    while (true) {
      const action = dqn.act(state);
      const { reward, observation, done, info } = dqn.env.step(action);
      state = observation;
      rewards += reward;
      if (episodeIteration > 60) {
        // after 60 episodes start rendering the evaluation to a web viewer to visualize the progress.
        await dqn.env.render('web', { fps: 60, episode: episodeIteration, rewards });
      }
      
      if (done) break;
    }
    console.log(
      `Episode ${episodeIteration} - Train Rewards: ${episodeRewards[episodeRewards.length - 1]} - Eval Rewards: ${rewards}`
    );
  };

  /**
   * train the policy and target networks using DQN with the following settings
   * 
   * This can take a while to train well
   */
  dqn.train({ totalEpisodes: 1000, batchSize: 128, verbose: true, epochCallback, });
};

main();
