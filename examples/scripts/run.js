/**
 * This script is the general way to run an environment and also render it to the web. This uses the CartPole environment as an example.
 */

const RL = require('../../lib');
const env = new RL.Environments.Examples.CartPole();

const main = async () => {
  // run 100 episodes
  for (let episode = 0; episode < 100; episode++) {
    let state = env.reset();
    // keep stepping forward through environment
    while (true) {
      // sample a random action and step forward with it
      const action = env.actionSpace.sample();
      const { reward, observation, done, info } = env.step(action);
      state = observation;
      // render the environment to a web viewer with 60 fps.
      // change 'web' to 'ansi' for terminal based rendering (which may not be available for all environments)
      await env.render('web', { fps: 60, episode });
      if (done) break;
    }
  }
};
main();
