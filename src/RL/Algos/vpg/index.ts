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
import nj, { NdArray } from 'numjs';
import { ActorCritic } from '../../Models/ac';
export interface VPGConfigs<Observation, Action> {
  /** Converts observations to tensors */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actor critic output tensor to tensor that works with environment. Necessary if in discrete action space! */
  actionToTensor: (action: tf.Tensor) => TensorLike;
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
  seed: number;
}
type Action = TensorLike;
type Observation = TensorLike;
export class VPG<
  ObservationSpace extends Space<Observation>,
  ActionSpace extends Space<Action>,
> extends Agent<Observation, Action> {
  public configs: VPGConfigs<Observation, Action> = {
    obsToTensor: (obs: Observation) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if observation space is dict. if observation space is dict, user needs to override this.
      const tensor = np.tensorLikeToTensor(obs);
      return tensor.reshape([1, ...tensor.shape]);
    },
    actionToTensor: (action: tf.Tensor) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if action space is dict. if action space is dict, user needs to override this.
      return action;
    }
  };

  public env: Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>;

  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: tf.Tensor) => TensorLike;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>,
    /** The actor crtic model */
    public ac: ActorCritic<tf.Tensor>,
    /** configs for the VPG model */
    configs: DeepPartial<VPGConfigs<Observation, Action>> = {}
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
    return np.tensorLikeToNdArray(this.ac.act(np.tensorLikeToTensor(observation)));
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
      seed: 0,
      stepCallback: () => {},
      epochCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);

    const { vf_optimizer, pi_optimizer } = configs;

    // TODO do some seeding things
    configs.seed += 99999 * ct.id();
    random.seed(configs.seed);
    // seed tensorflow
    
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
      return tf.tidy(() => {
        console.log(obs.shape)
        const { pi, logp_a } = this.ac.pi.apply(obs, act);
        const loss_pi = logp_a!.mul(adv).mean();

        let approx_kl = logp_old.sub(logp_a!).mean().arraySync();
        let ent = pi.entropy().mean().arraySync();
        return {
          loss_pi,
          pi_info: {
            approx_kl,
            ent
          }
        }
      });
    }
    const compute_loss_vf = (data: VPGBufferComputations) => {
      const {obs, ret} = data;
      return (this.ac.v.apply(obs).sub(ret)).pow(2).mean();
    }

    const update = async () => {
      const data = await buffer.get();
      const loss_pi_old = compute_loss_pi(data);
      // single gd step for policy.
      let pi_info_new: {
        approx_kl: number,
        ent: number
      };
      const pi_grads = pi_optimizer.computeGradients(() => {
        const {loss_pi, pi_info} = compute_loss_pi(data);
        pi_info_new = pi_info as typeof pi_info_new;
        return loss_pi as tf.Scalar;
      });
      // TODO: mpi avg grads here
      pi_optimizer.applyGradients(pi_grads.grads);

      for (let i = 0; i < configs.train_v_iters; i++) {
        const vf_grads = vf_optimizer.computeGradients(() => {
          const loss_v = compute_loss_vf(data);
          return loss_v as tf.Scalar; 
        });
        vf_optimizer.applyGradients(vf_grads.grads);
        // TODO: mpi avg grads here
      }
      // TODO
      // log changes
      console.log("=====")
      console.log("KL, ENT", pi_info_new!);
    }

    let start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_len = 0;
    for (let epoch = 0; epoch < configs.epochs; epoch++) {
      for (let t = 0; t < local_steps_per_epoch; t++) {
        // TODO
        let { a, v, logp_a } = this.ac.step(this.obsToTensor(o));
        let action = np.tensorLikeToNdArray(this.actionToTensor(a));
        let stepInfo = env.step(action);
        let next_o = stepInfo.observation;
        let r = stepInfo.reward;
        let d = stepInfo.done;
        ep_ret += 1;
        ep_len += 1;

        // TODO, log vvals
        buffer.store(np.unsqueeze(np.tensorLikeToNdArray(o), 0), np.tensorLikeToNdArray(a), r, np.tensorLikeToNdArray(v).get(0, 0), np.tensorLikeToNdArray(logp_a!).get(0, 0));

        o = next_o;

        const timeout = ep_len === configs.max_ep_len;
        const terminal = d || timeout;
        const epoch_ended = t === local_steps_per_epoch - 1;
        if (terminal || epoch_ended) {
          if (epoch_ended && !terminal) {
            console.log(`Warning: trajectory cut off by epoch at ${ep_len} steps`);
          }
          let v = 0;
          if (timeout || epoch_ended) {
            v = (this.ac.step(np.tensorLikeToTensor(o)).v.arraySync() as number[])[0];
          }
          buffer.finishPath(v);
          if (terminal) {
            // store ep ret and eplen stuff
          }
          o = env.reset();
          console.log({terminal, t, epoch_ended, epoch, ep_len, ep_ret})
          ep_ret = 0; ep_len = 0;
        }
      }
      // TODO save model

      // update actor critic
      await update();

      // log info about the epoch!?
    }
  }
}
