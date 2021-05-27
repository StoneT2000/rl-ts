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
policyNet.add(tf.layers.dense({units: 24, inputShape: [4], activation: 'tanh'}));
policyNet.add(tf.layers.dense({units: 48, activation: 'tanh'}));
policyNet.add(tf.layers.dense({units: 2, activation: 'linear' }))
const targetNet = tf.sequential();
targetNet.add(tf.layers.dense({units: 24, inputShape: [4], activation: 'tanh'}));
targetNet.add(tf.layers.dense({units: 48, activation: 'tanh'}));
targetNet.add(tf.layers.dense({units: 2, activation: 'linear' }))
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

const evaluateModel = (data: {time: number}) => {
  let state = dqn.env.reset();
  let reward = 0;
  while(true) {
    const stateTensor = np.toTensor(state).reshape([1, 4]);
    let action = (dqn.policyNet.predict(stateTensor) as tf.Tensor).argMax().dataSync()[0];
    const stepInfo = dqn.env.step(action);
    reward += stepInfo.reward;
    state = stepInfo.observation;
    if (stepInfo.done) break;
  }
  console.log(`Step ${data.time} eval: Rewards ${reward}`);
};

// dqn.train({
//   optimizer: tf.train.adam(1e-3),
//   batchSize: 128,
//   gamma: 0.999,
//   epsStart: 0.9,
//   epsEnd: 0.2,
//   learningStarts: 10000,
//   explorationTimeSteps: 10000,
//   totalTimeSteps: 100000,
//   policyTrainFreq: 1,
//   targetUpdateFreq: 10,
//   epochCallback: evaluateModel
// });

  let state = dqn.env.reset();
  let reward = 0;
  let dir = false;
  while(true) {
    const stateTensor = np.toTensor(state).reshape([1, 4]);
    // let action = (dqn.policyNet.predict(stateTensor) as tf.Tensor).argMax().dataSync()[0];
    let action = 0;
    if (dir) {
      action = 1;
    }
    dir = !dir;
    // if 
    // if ()
    console.log(state.selection.data)
    const stepInfo = dqn.env.step(action);
    state = stepInfo.observation;
    reward += stepInfo.reward;
    if (stepInfo.done) break;
  }
  console.log("Scored ", reward);



// let a = np.pack([[-0.05, -0.02, 0.01, 0.05]])
// let t = np.toTensor(a);
// let p = policyNet.predict(t) as tf.Tensor;
// console.log()
// console.log(p.dataSync())
// const a = tf.variable(tf.tensor1d([3, 4]));
// const b = tf.variable(tf.tensor1d([5, 6]));
// const x = np.toTensor(np.pack([[-0.05, -0.02, 0.01, 0.05], [0.05, 0.02, 0.01, 0.05], [-0.01, -0.04, 0.01, 0.05]]));

// const f = (): Scalar => {
//   console.log(x.shape)
//   let vals = (policyNet.predict(x) as tf.Tensor);
//   let vals2 = targetNet.predict(x) as tf.Tensor;
//   vals.print();
//   return tf.losses.huberLoss(vals2, vals.mul(3))
// }; // f is a function
// // df/da = x ^ 2, df/db = x 
// const {value, grads} = tf.variableGrads(f); // gradient of f as respect of each variable

// Object.keys(grads).forEach(varName => grads[varName].print());