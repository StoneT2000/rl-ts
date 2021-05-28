import { Agent } from '../../Agent';
import { Environment } from '../../Environments';
import { Space } from '../../Spaces';
import * as random from '../../utils/random';
import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer, Transition } from './replayBuffer';
import { DeepPartial } from '../../utils/types';
import { deepMerge } from '../../utils/deep';
import * as np from '../../utils/np';
import { Scalar } from '@tensorflow/tfjs';

export interface DQNConfigs<State, Action> {
  replayBufferCapacity: number;
  policyNet?: tf.LayersModel;
  targetNet?: tf.LayersModel;
  stateToTensor: (state: State) => tf.Tensor;
  actionToTensor: (action: Action) => tf.Tensor;
  /** Optional act function to replace the default act */
  act?: (state: State) => Action;
}
export interface DQNTrainConfigs<State, Action> {
  stepCallback(
    stepData: Transition<State, Action> & {
      time: number;
      episodeDurations: number[];
      episodeRewards: number[];
      episodeIteration: number;
      info: any;
    }
  ): any;
  epochCallback(epochDate: {
    time: number;
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
  ckptFreq: number;
  learningStarts: number;
  totalEpisodes: number;
  verbose: boolean;
  batchSize: number;
}

export class DQN<ObservationSpace extends Space<State>, ActionSpace extends Space<Action>, State, Action> extends Agent<
  State,
  Action
> {
  public configs: DQNConfigs<State, Action> = {
    replayBufferCapacity: 1000,
    stateToTensor: () => {
      throw new Error('stateToTensor function not provided');
    },
    actionToTensor: () => {
      throw new Error('actionToTensor function not provided');
    },
  };
  public replayBuffer: ReplayBuffer<State, Action>;

  public env: Environment<ObservationSpace, ActionSpace, State, Action, number>;

  public policyNet: tf.LayersModel;
  public targetNet: tf.LayersModel;

  private stateToTensor: (state: State) => tf.Tensor;
  private actionToTensor: (action: Action) => tf.Tensor;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, State, Action, number>,
    /** configs for the DQN model */
    configs: DeepPartial<DQNConfigs<State, Action>>
  ) {
    super();
    this.configs = deepMerge(this.configs, configs);

    this.env = makeEnv();
    if (!this.configs.policyNet || !this.configs.targetNet) {
      throw new Error('Policy net or target net not provided');
    }
    this.policyNet = this.configs.policyNet;
    this.targetNet = this.configs.targetNet;
    this.stateToTensor = this.configs.stateToTensor;
    this.actionToTensor = this.configs.actionToTensor;
    this.replayBuffer = new ReplayBuffer(this.configs.replayBufferCapacity);
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: State): Action {
    if (this.configs.act) return this.configs.act(observation);
    const pred = this.policyNet.predict(this.stateToTensor(observation), {}) as tf.Tensor;
    const action = np.fromTensorSync(pred.argMax(1)).get(0);
    return action as any;
  }

  public async train(trainConfigs: Partial<DQNTrainConfigs<State, Action>>) {
    let configs: DQNTrainConfigs<State, Action> = {
      optimizer: tf.train.adam(1e-3),
      gamma: 0.999,
      epsStart: 0.9,
      epsEnd: 0.05,
      epsDecay: 200,
      learningStarts: 0,
      policyTrainFreq: 1,
      targetUpdateFreq: 10,
      ckptFreq: 100,
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
      const stepInfo = this.env.step(action);

      // store next state and push to replay buffer
      const nextState = stepInfo.observation;
      const done = stepInfo.done;
      const reward = stepInfo.reward;
      this.replayBuffer.push({ state, nextState, reward, action, done });

      // Move to next state
      state = nextState;
      episodeDurations[episodeDurations.length - 1] += 1;
      episodeRewards[episodeRewards.length - 1] += reward;

      // perform a step of optimization on policy net
      if (t >= configs.learningStarts && t % configs.policyTrainFreq === 0) {
        await this.optimizeModel({
          gamma,
          batchSize: configs.batchSize,
          optimizer,
        });
      }

      if (t >= configs.learningStarts && t % configs.ckptFreq === 0) {
        // TODO
      }
      // TODO: allow user to specify async step callbacks
      await configs.stepCallback({
        state,
        nextState,
        reward,
        action,
        time: t,
        episodeRewards,
        episodeDurations,
        episodeIteration,
        info: stepInfo.info,
        done,
      });

      if (done) {
        // update target with policy net
        if (t >= configs.learningStarts && episodeIteration % configs.targetUpdateFreq === 0) {
          this.targetNet.setWeights(this.policyNet.getWeights());
        }

        await configs.epochCallback({
          time: t,
          episodeDurations,
          episodeRewards,
          episodeIteration,
        });

        episodeDurations.push(0);
        episodeRewards.push(0);
        episodeIteration = episodeIteration + 1;
        state = this.env.reset();
      }
    }
  }

  private actEps(obs: State, eps: number): Action {
    if (random.randomVal() > eps) {
      if (this.configs.act) return this.configs.act(obs);
      const pred = this.policyNet.predict(this.stateToTensor(obs)) as tf.Tensor;
      const action = np.fromTensorSync(pred.argMax(1)).get(0) as any;
      return action;
    } else {
      return this.env.actionSpace.sample() as $TSFIXME;
    }
  }

  /** vary epsilon by exponential schedule */
  private getEpsilon(timeStep: number, epsDecay: number, epsStart: number, epsEnd: number) {
    const eps_threshold = epsEnd + (epsStart - epsEnd) * Math.exp((-1 * timeStep) / epsDecay);
    return eps_threshold;
  }

  /**
   * Run a step of optimization
   */
  public async optimizeModel(configs: { optimizer: tf.Optimizer; batchSize: number; gamma: number }) {
    if (this.replayBuffer.length < configs.batchSize) {
      return;
    }
    const transitions = this.replayBuffer.sample(configs.batchSize);
    let loss = 0;
    tf.tidy(() => {
      const optimizer = configs.optimizer;

      // TODO: consider having user provide toTensor functions that handle batches?
      const _stateBatch: any[] = [];
      const _actionBatch: (tf.Tensor<tf.Rank> | tf.TensorLike)[] = [];
      const _nextStateBatch: any[] = [];
      const _rewardBatch: any[] = [];
      const _doneBatch: any[] = [];
      for (let i = 0; i < configs.batchSize; i++) {
        _stateBatch.push(this.stateToTensor(transitions[i].state));
        _actionBatch.push(this.actionToTensor(transitions[i].action));
        _rewardBatch.push(transitions[i].reward);
        _nextStateBatch.push(this.stateToTensor(transitions[i].nextState));
        _doneBatch.push(transitions[i].done);
      }

      const stateBatch = tf.concat(_stateBatch); // [B, D]
      const actionBatch = tf.concat(_actionBatch).asType('int32'); // [B]
      const rewardBatch = tf.concat(_rewardBatch); // [B]
      const nextStateBatch = tf.concat(_nextStateBatch); // [B, D]

      this.targetNet.trainable = false;
      const lossFunc = () => {
        const stateValues = this.policyNet.apply(stateBatch) as tf.Tensor; // [B, 2]
        // TODO: remove hardcoded shape here

        const stateActionValues = stateValues.mul(tf.oneHot(actionBatch, stateValues.shape[1]!)).sum(-1);
        const nnn = this.targetNet.apply(nextStateBatch) as tf.Tensor;
        const nextStateValues = nnn.max(1);
        const doneMask = tf.scalar(1).sub(tf.tensor1d(_doneBatch).asType('float32')); // mask with 1s where there exists a next state and 0s where it is finished and no next state
        const expectedStateActionValues = nextStateValues.mul(doneMask).mul(configs.gamma).add(rewardBatch);

        const loss = tf.losses.huberLoss(stateActionValues, expectedStateActionValues) as Scalar;
        return loss as Scalar;
      };
      const grads = tf.variableGrads(lossFunc);
      for (const key of Object.keys(grads.grads)) {
        grads.grads[key] = grads.grads[key].clipByValue(-1, 1);
      }
      optimizer.applyGradients(grads.grads);
      this.targetNet.trainable = true;
      loss = grads.value.dataSync()[0];
    });
    return loss;
  }
}
