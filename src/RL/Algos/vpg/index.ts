import { Agent } from '../../Agent';
import { Environment } from '../../Environments';
import { Space } from '../../Spaces';
import * as random from '../../utils/random';
import * as tf from '@tensorflow/tfjs';
import { VPGBuffer, VPGBufferComputations } from './buffer';
import { DeepPartial } from '../../utils/types';
import { deepMerge } from '../../utils/deep';
import * as np from '../../utils/np';
import * as ct from '../../utils/clusterTools';
import { ActorCritic } from '../utils/models';
export interface VPGConfigs<Observation extends tf.Tensor, Action> {
  /** Converts observations to tensors that can be used in optimization function */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actions to tensors that can be used in optimization function */
  actionToTensor: (action: Action) => tf.Tensor;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface VPGTrainConfigs {
  stepCallback(
    stepData: {
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
  pi_optimizer: tf.Optimizer;
  vf_optimizer: tf.Optimizer;
  /** How frequently in terms of total steps to save the model. This is not used if saveDirectory is not provided */
  ckptFreq: number;
  /** path to store saved models in. */
  savePath?: string;
  saveLocation?: TFSaveLocations;
  epochs: number;
  verbose: boolean;
  gamma: number;
  lam: number;
  train_v_iters: number;
  steps_per_epoch: number;
  /** maximum length of each trajectory collected */
  max_ep_len: number;
}
type Action = tf.Tensor;
export class VPG<
  ObservationSpace extends Space<Observation>,
  ActionSpace extends Space<Action>,
  Observation extends tf.Tensor
> extends Agent<Observation, Action> {
  public configs: VPGConfigs<Observation, Action> = {
    obsToTensor: (obs: Observation) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if observation space is dict. if observation space is dict, user needs to override this.
      const tensor = np.tensorLikeToTensor(obs);
      return tensor.reshape([1, ...tensor.shape]);
    },
    actionToTensor: (action: Action) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if action space is dict. if action space is dict, user needs to override this.
      return np.tensorLikeToTensor(action);
    }
  };

  public env: Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>;

  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: Action) => tf.Tensor;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>,
    /** The actor crtic model */
    public ac: ActorCritic<Observation>,
    /** configs for the VPG model */
    configs: DeepPartial<VPGConfigs<Observation, Action>>
  ) {
    super();
    this.configs = deepMerge(this.configs, configs);

    this.env = makeEnv();
    if (!this.env.actionSpace.meta.discrete) throw new Error('Action space is not discrete');
    this.obsToTensor = this.configs.obsToTensor;
    this.actionToTensor = this.configs.actionToTensor;
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: Observation): Action {
    return this.ac.act(observation) as tf.Tensor;
  }

  public async train(trainConfigs: Partial<VPGTrainConfigs>) {
    let configs: VPGTrainConfigs = {
      vf_optimizer: tf.train.adam(1e-3),
      pi_optimizer: tf.train.adam(1e-3),
      ckptFreq: 1000,
      verbose: false,
      steps_per_epoch: 10000,
      max_ep_len: 1000,
      epochs: 50,
      train_v_iters: 80,
      gamma: 0.99,
      lam: 0.97,
      stepCallback: () => {},
      epochCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);

    // TODO do some seeding things

    let env = this.env;
    let obs_dim = env.observationSpace.shape;
    let act_dim = env.actionSpace.shape;

    let local_steps_per_epoch = configs.steps_per_epoch / ct.numProcs();

    const buffer = new VPGBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: act_dim,
      obsDim: obs_dim,
      size: local_steps_per_epoch
    });

    if (configs.verbose) {
      console.log('Beginning training with configs', configs);
    }

    const compute_loss_pi = (data: VPGBufferComputations) => {
      const {obs, act, adv} = data;
      const logp_old = data.logp;
      // TODO:
    }
    const compute_loss_vf = (data: VPGBufferComputations) => {
      // TODO:
    }

    const update = () => {
      const data = buffer.get();
      // TODO
    }

    let start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_len = 0;
    for (let epoch = 0; epoch < configs.epochs; epoch++) {
      for (let t = 0; t < local_steps_per_epoch; t++) {
        // TODO
      }
    }
  }
}
