import { Agent } from 'rl-ts/lib/Agent';
import { Environment } from 'rl-ts/lib/Environments';
import { Space } from 'rl-ts/lib/Spaces';
import * as random from 'rl-ts/lib/utils/random';
import * as tf from '@tensorflow/tfjs';
import { VPGBuffer, VPGBufferComputations } from 'rl-ts/lib/Algos/vpg/buffer';
import { DeepPartial } from 'rl-ts/lib/utils/types';
import { deepMerge } from 'rl-ts/lib/utils/deep';
import * as np from 'rl-ts/lib/utils/np';
import * as ct from 'rl-ts/lib/utils/clusterTools';
import { NdArray } from 'numjs';
import { ActorCritic } from 'rl-ts/lib/Models/ac';
import pino from 'pino';
const log = pino({
  prettyPrint: {
    colorize: true,
  },
});

export interface VPGConfigs<Observation, Action> {
  /** Converts observations to tensors */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actor critic output tensor to tensor that works with environment. Necessary if in discrete action space! */
  actionToTensor: (action: tf.Tensor) => TensorLike;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface VPGTrainConfigs {
  stepCallback(stepData: {
    step: number;
    episodeDurations: number[];
    episodeRewards: number[];
    episodeIteration: number;
    info: any;
    loss: number | null;
  }): any;
  epochCallback(epochDate: {
    epoch: number;
    kl: number;
    entropy: number;
    delta_pi_loss: number;
    delta_vf_loss: number;
    ep_rets: {
      mean: number,
      std: number,
    }
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
  train_pi_iters: number;
  steps_per_epoch: number;
  /** maximum length of each trajectory collected */
  max_ep_len: number;
  seed: number;
  name: string;
}
type Action = NdArray | number;
type Observation = NdArray;
export class VPG<ObservationSpace extends Space<Observation>, ActionSpace extends Space<Action>> extends Agent<
  Observation,
  Action
> {
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
    },
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
    return np.tensorLikeToNdArray(this.actionToTensor(this.ac.act(this.obsToTensor(observation))));
  }

  public async train(trainConfigs: Partial<VPGTrainConfigs>) {
    let configs: VPGTrainConfigs = {
      vf_optimizer: tf.train.adam(1e-3),
      pi_optimizer: tf.train.adam(3e-4),
      ckptFreq: 1000,
      verbose: false,
      steps_per_epoch: 10000,
      max_ep_len: 1000,
      epochs: 50,
      train_v_iters: 80,
      gamma: 0.99,
      lam: 0.97,
      seed: 0,
      train_pi_iters: 1,
      name: 'VPG_Train',
      stepCallback: () => {},
      epochCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);

    const { vf_optimizer, pi_optimizer } = configs;

    // TODO do some seeding things
    configs.seed += 99999 * ct.id();
    random.seed(configs.seed);
    // TODO: seed tensorflow if possible

    const env = this.env;
    const obs_dim = env.observationSpace.shape;
    const act_dim = env.actionSpace.shape;

    let local_steps_per_epoch = configs.steps_per_epoch / ct.numProcs();
    if (Math.ceil(local_steps_per_epoch) !== local_steps_per_epoch) {
      configs.steps_per_epoch = Math.ceil(local_steps_per_epoch) * ct.numProcs();
      log.warn(
        `${configs.name} | Changing steps per epoch to ${
          configs.steps_per_epoch
        } as there are ${ct.numProcs()} processes running`
      );
      local_steps_per_epoch = configs.steps_per_epoch / ct.numProcs();
    }

    const buffer = new VPGBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: act_dim,
      obsDim: obs_dim,
      size: local_steps_per_epoch,
    });

    if (configs.verbose && ct.id() === 0) {
      log.info(configs, `${configs.name} | Beginning training with configs`);
    }

    type pi_info = {
      approx_kl: number;
      entropy: number;
    };
    const compute_loss_pi = (data: VPGBufferComputations): { loss_pi: tf.Tensor; pi_info: pi_info } => {
      const { obs, act, adv } = data;
      const logp_old = data.logp;
      return tf.tidy(() => {
        const { pi, logp_a } = this.ac.pi.apply(obs, act);
        // pi, logp_a, logp_old are all of shape [B]
        const loss_pi = logp_a!.mul(adv).mean().mul(-1);

        const approx_kl = logp_old.sub(logp_a!).mean().arraySync() as number;
        const entropy = pi.entropy().mean().arraySync() as number;
        return {
          loss_pi,
          pi_info: {
            approx_kl,
            entropy,
          },
        };
      });
    };
    const compute_loss_vf = (data: VPGBufferComputations) => {
      const { obs, ret } = data;
      return this.ac.v.apply(obs).sub(ret).pow(2).mean();
    };

    const update = async () => {
      const data = await buffer.get();
      const loss_pi_old = compute_loss_pi(data).loss_pi;
      const loss_vf_old = compute_loss_vf(data);
      let kl = 0;
      let entropy = 0;
      let loss_pi_new = 0;
      let loss_vf_new = 0;

      for (let i = 0; i < configs.train_pi_iters; i++) {
        const pi_grads = pi_optimizer.computeGradients(() => {
          const { loss_pi, pi_info } = compute_loss_pi(data);
          kl = pi_info.approx_kl;
          entropy = pi_info.entropy;
          return loss_pi as tf.Scalar;
        });
        await ct.averageGradients(pi_grads.grads);
        pi_optimizer.applyGradients(pi_grads.grads);
        if (i === configs.train_pi_iters - 1) {
          loss_pi_new = pi_grads.value.arraySync();
        }
      }

      kl = await ct.avgNumber(kl);
      entropy = await ct.avgNumber(entropy);

      for (let i = 0; i < configs.train_v_iters; i++) {
        const vf_grads = vf_optimizer.computeGradients(() => {
          const loss_v = compute_loss_vf(data);
          return loss_v as tf.Scalar;
        });
        await ct.averageGradients(vf_grads.grads);
        vf_optimizer.applyGradients(vf_grads.grads);
        if (i === configs.train_pi_iters - 1) {
          loss_vf_new = vf_grads.value.arraySync();
        }
      }
      let delta_pi_loss = loss_pi_new - (loss_pi_old.arraySync() as number);
      let delta_vf_loss = loss_vf_new - (loss_vf_old.arraySync() as number);
      delta_pi_loss = await ct.avgNumber(delta_pi_loss);
      delta_vf_loss = await ct.avgNumber(delta_vf_loss);
      const metrics = { kl, entropy, delta_pi_loss, delta_vf_loss };
      return metrics;
    };

    // const start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_len = 0;
    let ep_rets = [];
    for (let epoch = 0; epoch < configs.epochs; epoch++) {
      for (let t = 0; t < local_steps_per_epoch; t++) {
        // TODO
        const { a, v, logp_a } = this.ac.step(this.obsToTensor(o));
        const action = np.tensorLikeToNdArray(this.actionToTensor(a));
        const stepInfo = env.step(action);
        const next_o = stepInfo.observation;

        const r = stepInfo.reward;
        const d = stepInfo.done;
        ep_ret += r;
        ep_len += 1;

        // TODO, log vvals
        buffer.store(
          np.unsqueeze(np.tensorLikeToNdArray(o), 0),
          np.tensorLikeToNdArray(a),
          r,
          np.tensorLikeToNdArray(v).get(0, 0),
          np.tensorLikeToNdArray(logp_a!).get(0, 0)
        );

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
            v = (this.ac.step(this.obsToTensor(o)).v.arraySync() as number[][])[0][0];
          }
          buffer.finishPath(v);
          if (terminal) {
            // store ep ret and eplen stuff
            ep_rets.push(ep_ret);
          }
          o = env.reset();
          ep_ret = 0;
          ep_len = 0;
        }
      }
      // TODO save model

      // update actor critic
      const metrics = await update();
      const ep_rets_metrics = await ct.statisticsScalar(np.tensorLikeToTensor(ep_rets));

      if (ct.id() === 0) {
        const msg = `${configs.name} - Epoch ${epoch} metrics: `;
        log.info(new Array(msg.length + 1).join('='));
        log.info(
          {
            ...metrics,
            ep_rets: ep_rets_metrics
          },
          msg
        );
      }
      await configs.epochCallback({
        epoch,
        ...metrics,
        ep_rets: ep_rets_metrics
      });

      ep_rets = [];
    }
  }
}
