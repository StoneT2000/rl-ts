import { Agent } from 'rl-ts/lib/Agent';
import { Environment } from 'rl-ts/lib/Environments';
import { Space } from 'rl-ts/lib/Spaces';
import * as random from 'rl-ts/lib/utils/random';
import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer, Transition } from './replayBuffer';
import { DeepPartial } from 'rl-ts/lib/utils/types';
import { deepMerge } from 'rl-ts/lib/utils/deep';
import * as np from 'rl-ts/lib/utils/np';
import { Scalar } from '@tensorflow/tfjs';
import { NdArray } from 'numjs';


export interface DQNConfigs<Observation, Action> {
  replayBufferCapacity: number;
  policyNet?: tf.LayersModel;
  targetNet?: tf.LayersModel;
  /** Converts observations to tensors with a batch size dimension that can be used in optimization function */
  obsToTensor: (obs: Observation) => tf.Tensor;
  /** Converts actions to tensors that can be used in optimization function */
  actionToTensor: (action: Action) => tf.Tensor;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface DQNTrainConfigs<Observation, Action> {
  stepCallback(
    stepData: Transition<Observation, Action> & {
      step: number;
      episodeDurations: number[];
      episodeRewards: number[];
      episodeIteration: number;
      info: any;
      loss: number | null;
    }
  ): any;
  epochCallback(epochDate: {
    step: number;
    episodeDurations: number[];
    episodeRewards: number[];
    episodeIteration: number;
  }): any;
  optimizer: tf.Optimizer;
  gamma: number;
  epsStart: number;
  epsEnd: number;
  epsDecay: number;
  policyTrainFreq: number;
  targetUpdateFreq: number;
  /** How frequently in terms of total steps to save the model. This is not used if saveDirectory is not provided */
  ckptFreq: number;
  /** path to store saved models in. */
  savePath?: string;
  saveLocation?: TFSaveLocations;
  learningStarts: number;
  totalEpisodes: number;
  verbose: boolean;
  batchSize: number;
}
type Observation = NdArray;
type Action = NdArray | number;
export class DQN<ObservationSpace extends Space<Observation>, ActionSpace extends Space<Action>> extends Agent<
  Observation,
  Action
> {
  public configs: DQNConfigs<Observation, Action> = {
    replayBufferCapacity: 1000,
    obsToTensor: (obs: Observation) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if observation space is dict. if observation space is dict, user needs to override this.
      const tensor = np.tensorLikeToTensor(obs);
      if (tensor.shape[0] !== this.env.observationSpace.shape[0]) {
        // if there is a mismatch in first dimension, we assume this is the batch size
        return tensor;
      }
      return tensor.reshape([1, ...tensor.shape]);
    },
    actionToTensor: (action: Action) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if action space is dict. if action space is dict, user needs to override this.
      return np.tensorLikeToTensor(action);
    },
  };
  public replayBuffer: ReplayBuffer<Observation, Action>;

  public env: Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>;

  public policyNet: tf.LayersModel;
  public targetNet: tf.LayersModel;

  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: Action) => tf.Tensor;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>,
    /** configs for the DQN algorithm */
    configs: DeepPartial<DQNConfigs<Observation, Action>>
  ) {
    super();
    this.configs = deepMerge(this.configs, configs);

    this.env = makeEnv();
    if (!this.configs.policyNet || !this.configs.targetNet) {
      throw new Error('Policy net or target net not provided');
    }
    if (!this.env.actionSpace.meta.discrete) throw new Error('Action space is not discrete');
    this.policyNet = this.configs.policyNet;
    this.targetNet = this.configs.targetNet;
    this.obsToTensor = this.configs.obsToTensor;
    this.actionToTensor = this.configs.actionToTensor;
    this.replayBuffer = new ReplayBuffer(this.configs.replayBufferCapacity);
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: Observation): Action {
    if (this.configs.act) return this.configs.act(observation);
    const pred = this.policyNet.predict(this.obsToTensor(observation), {}) as tf.Tensor;
    const action = np.tensorLikeToNdArray(pred.argMax(1)).get(0);
    return action as any;
  }

  public async train(trainConfigs: Partial<DQNTrainConfigs<Observation, Action>>) {
    let configs: DQNTrainConfigs<Observation, Action> = {
      optimizer: tf.train.adam(1e-3),
      gamma: 0.999,
      epsStart: 0.9,
      epsEnd: 0.05,
      epsDecay: 200,
      learningStarts: 0,
      policyTrainFreq: 1,
      targetUpdateFreq: 10,
      ckptFreq: 1000,
      totalEpisodes: 200,
      verbose: false,
      batchSize: 32,
      stepCallback: () => {},
      epochCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);

    if (configs.verbose) {
      console.log('Beginning training with configs', configs);
    }
    const { optimizer, gamma } = configs;
    let state = this.env.reset();
    const episodeDurations = [0];
    const episodeRewards = [0.0];
    let episodeIteration = 0;

    for (let t = 0; episodeIteration < configs.totalEpisodes; t++) {
      // Select and perform an action
      const eps = this.getEpsilon(t, configs.epsDecay, configs.epsStart, configs.epsEnd);
      const action = this.actEps(state, eps);

      // Step through environment
      const stepInfo = this.env.step(action);

      // store next state and push to replay buffer
      const nextState = stepInfo.observation;
      const done = stepInfo.done;
      const reward = stepInfo.reward;
      this.replayBuffer.push({ state, nextState, reward, action, done });

      // Move to next state and store rewards and durations
      state = nextState;
      episodeDurations[episodeDurations.length - 1] += 1;
      episodeRewards[episodeRewards.length - 1] += reward;

      // perform a step of optimization on policy net
      let loss: number | null = null;
      if (t >= configs.learningStarts && t % configs.policyTrainFreq === 0) {
        loss = await this.optimizeModel({
          gamma,
          batchSize: configs.batchSize,
          optimizer,
        });
      }
      if (configs.savePath && configs.saveLocation) {
        if (t >= configs.learningStarts && t % configs.ckptFreq === 0) {
          // save policy and target net models.
          const policyNetSavePath = `${configs.saveLocation}://${configs.savePath}/policynet-${t}`;
          const targetNetSavePath = `${configs.saveLocation}://${configs.savePath}/targetnet-${t}`;
          if (configs.verbose) {
            console.log(`Saving policy and target networks to ${policyNetSavePath} and ${targetNetSavePath}`);
          }
          this.policyNet.save(policyNetSavePath);
          this.targetNet.save(targetNetSavePath);
        }
      }

      await configs.stepCallback({
        state,
        nextState,
        reward,
        action,
        step: t,
        episodeRewards,
        episodeDurations,
        episodeIteration,
        info: stepInfo.info,
        done,
        loss,
      });

      if (done) {
        // update target with policy net
        if (t >= configs.learningStarts && episodeIteration % configs.targetUpdateFreq === 0) {
          this.targetNet.setWeights(this.policyNet.getWeights());
        }

        await configs.epochCallback({
          step: t,
          episodeDurations,
          episodeRewards,
          episodeIteration,
        });

        // setup next episode
        episodeDurations.push(0);
        episodeRewards.push(0);
        episodeIteration = episodeIteration + 1;
        state = this.env.reset();
      }
    }
  }

  /**
   * Select action with eps probability of sampling from action space
   * @param obs - observation
   * @param eps - probability of selecting random action
   * @returns
   */
  private actEps(obs: Observation, eps: number): Action {
    if (random.randomVal() > eps) {
      if (this.configs.act) return this.configs.act(obs);
      const pred = this.policyNet.predict(this.obsToTensor(obs)) as tf.Tensor;
      const action = np.fromTensorSync(pred.argMax(1)).get(0) as $TSFIXME;
      return action;
    } else {
      // TODO: allow user to provide their own sampling algorithm
      return this.env.actionSpace.sample();
    }
  }

  // TODO: allow user to use other schedules or define their own
  /** vary epsilon by exponential schedule */
  private getEpsilon(timeStep: number, epsDecay: number, epsStart: number, epsEnd: number) {
    const eps_threshold = epsEnd + (epsStart - epsEnd) * Math.exp((-1 * timeStep) / epsDecay);
    return eps_threshold;
  }

  /**
   * Run a step of optimization
   */
  public async optimizeModel(configs: {
    optimizer: tf.Optimizer;
    batchSize: number;
    gamma: number;
  }): Promise<number | null> {
    if (this.replayBuffer.length < configs.batchSize) {
      return null;
    }
    const transitions = this.replayBuffer.sample(configs.batchSize);

    const loss = tf.tidy(() => {
      const optimizer = configs.optimizer;
      const _stateBatch: any[] = [];
      const _actionBatch: (tf.Tensor<tf.Rank> | tf.TensorLike)[] = [];
      const _nextStateBatch: any[] = [];
      const _rewardBatch: any[] = [];
      const _doneBatch: any[] = [];
      for (let i = 0; i < configs.batchSize; i++) {
        _stateBatch.push(this.obsToTensor(transitions[i].state));
        _actionBatch.push(this.actionToTensor(transitions[i].action));
        _rewardBatch.push(transitions[i].reward);
        _nextStateBatch.push(this.obsToTensor(transitions[i].nextState));
        _doneBatch.push(transitions[i].done);
      }

      const stateBatch = tf.concat(_stateBatch); // [B, D]
      const actionBatch = tf.concat(_actionBatch).asType('int32'); // [B]
      const rewardBatch = tf.concat(_rewardBatch); // [B]
      const nextStateBatch = tf.concat(_nextStateBatch); // [B, D]

      this.targetNet.trainable = false;
      const lossFunc = () => {
        const stateValues = this.policyNet.apply(stateBatch) as tf.Tensor; // [B, actdim]

        const stateActionValues = stateValues.mul(tf.oneHot(actionBatch, stateValues.shape[1]!)).sum(-1); // this zeros out state values associated to unselected actions
        const nnn = this.targetNet.apply(nextStateBatch) as tf.Tensor;
        const nextStateValues = nnn.max(1);
        const doneMask = tf.scalar(1).sub(tf.tensor1d(_doneBatch).asType('float32')); // mask with 1s where there exists a next state and 0s where it is finished and no next state
        const expectedStateActionValues = nextStateValues.mul(doneMask).mul(configs.gamma).add(rewardBatch);

        const loss = tf.losses.huberLoss(stateActionValues, expectedStateActionValues) as Scalar;
        return loss as Scalar;
      };
      const grads = optimizer.computeGradients(lossFunc);
      for (const key of Object.keys(grads.grads)) {
        grads.grads[key] = grads.grads[key].clipByValue(-1, 1);
      }
      optimizer.applyGradients(grads.grads);
      this.targetNet.trainable = true;
      return grads.value.dataSync()[0];
    });
    return loss;
  }
}
