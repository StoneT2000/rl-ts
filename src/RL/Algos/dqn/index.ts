import { Agent } from '../../Agent';
import { Environment } from '../../Environments';
import { Space } from '../../Spaces';
import * as random from '../../utils/random';
import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer, Transition } from './replayBuffer';
import { DeepPartial } from '../../utils/types';
import { deepMerge } from '../../utils/deep';
import { NdArray } from 'ndarray';
import nj from 'numjs';
import * as np from "../../utils/np";
import { Scalar, Sequential } from '@tensorflow/tfjs';

export interface DQNConfigs<State, Action> {
  replayBufferCapacity: number;
  policyNet?: tf.LayersModel;
  targetNet?: tf.LayersModel;
  stateToTensor: (state: State) => tf.Tensor;
  actionToTensor: (action: Action) => tf.Tensor;
}
export interface DQNTrainConfigs<State, Action> {
  stepCallback(stepData: Transition<State, Action> & {
    time: number,
    episodeDurations: number[],
    episodeRewards: number[],
    info: any,
  }): any;
  epochCallback(epochDate: {
    time: number,
    episodeDurations: number[],
    episodeRewards: number[],
  }): any;
  optimizer: tf.Optimizer;
  gamma: number;
  epsStart: number;
  epsEnd: number;
  explorationTimeSteps: number;
  policyTrainFreq: number;
  targetUpdateFreq: number;
  ckptFreq: number;
  learningStarts: number;
  totalTimeSteps: number;
  verbose: boolean;
  batchSize: number,
}

export class DQN<ObservationSpace extends Space<State>, ActionSpace extends Space<Action>, State, Action> extends Agent<
  State,
  Action
> {
  public configs: DQNConfigs<State, Action> = {
    replayBufferCapacity: 1000,
    stateToTensor: () => { throw new Error("stateToTensor function not provided")},
    actionToTensor: () => { throw new Error("actionToTensor function not provided")}
  }
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
      throw new Error("Policy net or target net not provided");
    }
    this.policyNet = this.configs.policyNet;
    this.targetNet = this.configs.targetNet;
    this.stateToTensor = this.configs.stateToTensor;
    this.actionToTensor = this.configs.actionToTensor;
    this.replayBuffer = new ReplayBuffer(this.configs.replayBufferCapacity);
  };
  act(observation: State): Action {
    throw new Error('Method not implemented.');
  }

  public async train(trainConfigs: Partial<DQNTrainConfigs<State, Action>>) {
    let configs: DQNTrainConfigs<State, Action> = {
      optimizer: tf.train.adam(1e-3),
      gamma: 0.99,
      epsStart: 0.9,
      epsEnd: 0.15,
      explorationTimeSteps: 100,
      learningStarts: 100,
      policyTrainFreq: 1,
      targetUpdateFreq: 4,
      ckptFreq: 1000,
      totalTimeSteps: 10000,
      verbose: false,
      batchSize: 32,
      stepCallback: () => {},
      epochCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);

    if (configs.verbose)  {
      console.log("Beginning training with configs", configs);
    }
    const { optimizer, gamma } = configs;
    let state = this.env.reset();
    let episodeDurations = [0];
    let episodeRewards = [0.0];
    for (let t = 0; t < configs.totalTimeSteps; t++) {

      // Select and perform an action
      const eps = this.getEpsilon(t, configs.explorationTimeSteps, configs.epsStart, configs.epsEnd);
      const action = this.actEps(state, eps);
      //@ts-ignore
      const stepInfo = this.env.step(action);
      
      const nextState = stepInfo.observation;
      const done = stepInfo.done;
      const reward = stepInfo.reward;
      this.replayBuffer.push({ state, nextState, reward, action, done });

      // Move to next state
      state = nextState;
      episodeDurations[episodeDurations.length - 1] += 1;
      episodeRewards[episodeRewards.length - 1] += reward;

      // perform a step of optimization on policy net
      if (t > configs.learningStarts && t % configs.policyTrainFreq === 0) {
        await this.optimizeModel({
          gamma,
          batchSize: configs.batchSize,
          optimizer,
        });
      }

      // update target with policy net
      if (t > configs.learningStarts && t % configs.targetUpdateFreq === 0) {
        this.targetNet.setWeights(this.policyNet.getWeights());
      }

      if (t > configs.learningStarts && t % configs.ckptFreq === 0) {
        // TODO
      }
      // TODO: allow user to specify async step callbacks
      await configs.stepCallback({state, nextState, reward, action, time: t, episodeRewards, episodeDurations, info: stepInfo.info, done,});
      
      if (done) {
        // TODO: allow user to specify async epoch callbacks
        // console.log(`Episode ${episodeRewards.length} - rewards: ${episodeRewards[episodeRewards.length - 1]}, size ${this.replayBuffer.memory.length}`);
        configs.epochCallback({
          time: t,
          episodeDurations,
          episodeRewards
        });

        episodeDurations.push(0);
        episodeRewards.push(0);
        state = this.env.reset();
      }

    };
  }

  private actEps(obs: State, eps: number): Action {
    // return 0 as $TSFIXME;
    // return this.env.actionSpace.sample() as $TSFIXME;
    if (random.randomVal() > eps) {
      return np.fromTensorSync((this.policyNet.predict(this.stateToTensor(obs), {}) as tf.Tensor).argMax()).get(0) as $TSFIXME;
    } else {
      return this.env.actionSpace.sample() as $TSFIXME;
    }
  }

  /** vary epsilon by linear schedule */
  private getEpsilon(timeStep: number, explorationTimeSteps: number, epsStart: number, epsEnd: number) {
    const fraction = Math.min(timeStep / explorationTimeSteps, 1.0);
    return epsStart + fraction * (epsEnd - epsStart);
  };

  /**
   * Run a step of optimization
   */
  public async optimizeModel(configs: {
    optimizer: tf.Optimizer;
    batchSize: number;
    gamma: number;
  }) {
    if (this.replayBuffer.length < configs.batchSize) {
      return;
    }
    const transitions = this.replayBuffer.sample(configs.batchSize);
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
      let stateBatch = tf.concat(_stateBatch); // [B, D]
      const actionBatch = tf.concat(_actionBatch).asType("int32"); // [B]
      const rewardBatch = tf.concat(_rewardBatch); // [B]
      const nextStateBatch = tf.concat(_nextStateBatch); // [B, D]
      const doneBatch = tf.concat(_doneBatch);
    
      this.targetNet.trainable = false;
      // let x = np.toTensor(np.pack([[-0.05, -0.02, 0.01, 0.05], [0.05, 0.02, 0.01, 0.05], [-0.01, -0.04, 0.01, 0.05]]));
      const lossFunc = () => {

        let stateValues = (this.policyNet.apply(stateBatch) as tf.Tensor) // [B, 2]
        // TODO: remove hardcoded shape here
        const stateActionValues = stateValues.mul(tf.oneHot(actionBatch, 2)).sum(-1)

        const doneMask = tf.scalar(1).sub(
          tf.tensor1d(_doneBatch).asType('float32'));
        let nextStateValues = (this.targetNet.apply(nextStateBatch) as tf.Tensor).max(1);
        let expectedStateActionValues = nextStateValues.mul(doneMask).mul(configs.gamma).add(rewardBatch);

        const loss = tf.losses.huberLoss(stateActionValues, expectedStateActionValues) as Scalar;
        return loss as Scalar;
      }
      const grads = tf.variableGrads(lossFunc);
      for (const key of Object.keys(grads.grads)) {
        grads.grads[key] = grads.grads[key].clipByValue(-10, 10);
      }
      
      optimizer.applyGradients(grads.grads);
      this.targetNet.trainable = true;
    });
    return;
  }

}
