import { Agent } from 'rl-ts/lib/Agent';
import { Environment } from 'rl-ts/lib/Environments';
import { Discrete, Space } from 'rl-ts/lib/Spaces';
import * as random from 'rl-ts/lib/utils/random';
import * as tf from '@tensorflow/tfjs';
import { PPOBuffer, PPOBufferComputations } from 'rl-ts/lib/Algos/ppo/buffer';
import { DeepPartial } from 'rl-ts/lib/utils/types';
import { deepMerge } from 'rl-ts/lib/utils/deep';
import * as np from 'rl-ts/lib/utils/np';
import * as ct from 'rl-ts/lib/utils/clusterTools';
import { NdArray } from 'numjs';
import { ActorCritic } from 'rl-ts/lib/Models/ac';
import pino from 'pino';
import nj from 'numjs';
const log = pino({
  prettyPrint: {
    colorize: true,
  },
});

export interface PPOConfigs<Observation, Action> {
  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actor critic output tensor to tensor that works with environment. Necessary if in discrete action space! */
  actionToTensor: (action: tf.Tensor) => TensorLike;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface PPOTrainConfigs {
  stepCallback(stepData: {
    step: number;
    episodeDurations: number[];
    episodeRewards: number[];
    episodeIteration: number;
    info: any;
    loss: number | null;
  }): any;
  iterationCallback(iterationData: {
    iteration: number;
    kl: number;
    entropy: number;
    ep_rets: {
      mean: number;
      std: number;
    };
  }): any;
  optimizer: tf.Optimizer;
  /** How frequently in terms of total steps to save the model. This is not used if saveDirectory is not provided */
  ckptFreq: number;
  /** path to store saved models in. */
  savePath?: string;
  saveLocation?: TFSaveLocations;
  iterations: number;
  n_epochs: number;
  batch_size: number;
  vf_coef: number;
  verbosity: string;
  gamma: number;
  lam: number;
  target_kl: number;
  clip_ratio: number;
  steps_per_iteration: number;
  /** maximum length of each trajectory collected */
  max_ep_len: number;
  seed: number;
  name: string;
}
type Action = NdArray | number;
export class PPO<
  Observation,
  ObservationSpace extends Space<Observation>,
  ActionSpace extends Space<Action>
> extends Agent<Observation, Action> {
  public configs: PPOConfigs<Observation, Action> = {
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

  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: tf.Tensor) => TensorLike;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>,
    /** The actor crtic model */
    public ac: ActorCritic<tf.Tensor>,
    /** configs for the PPO model */
    configs: DeepPartial<PPOConfigs<Observation, Action>> = {}
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

  public async train(trainConfigs: Partial<PPOTrainConfigs>) {
    let configs: PPOTrainConfigs = {
      optimizer: tf.train.adam(3e-4),
      ckptFreq: 1000,
      steps_per_iteration: 10000,
      max_ep_len: 1000,
      iterations: 50,
      n_epochs: 10,
      batch_size: 64,
      vf_coef: 0.5,
      gamma: 0.99,
      lam: 0.97,
      clip_ratio: 0.2,
      seed: 0,
      target_kl: 0.01,
      verbosity: 'info',
      name: 'PPO_Train',
      stepCallback: () => {},
      iterationCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);
    log.level = configs.verbosity;

    const { clip_ratio, optimizer, target_kl } = configs;

    // TODO do some seeding things
    configs.seed += 99999 * ct.id();
    random.seed(configs.seed);
    // TODO: seed tensorflow if possible

    const env = this.env;
    const obs_dim = env.observationSpace.shape;
    let act_dim = env.actionSpace.shape;
    // store single value for discrete action space since we are using categorical distribution
    if (env.actionSpace instanceof Discrete) {
      act_dim = [1];
    }

    let local_steps_per_epoch = configs.steps_per_iteration / ct.numProcs();
    if (Math.ceil(local_steps_per_epoch) !== local_steps_per_epoch) {
      configs.steps_per_iteration = Math.ceil(local_steps_per_epoch) * ct.numProcs();
      log.warn(
        `${configs.name} | Changing steps per epoch to ${
          configs.steps_per_iteration
        } as there are ${ct.numProcs()} processes running`
      );
      local_steps_per_epoch = configs.steps_per_iteration / ct.numProcs();
    }

    const buffer = new PPOBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: act_dim,
      obsDim: obs_dim,
      size: local_steps_per_epoch,
    });

    if (ct.id() === 0) {
      log.info(configs, `${configs.name} | Beginning training with configs`);
    }

    type pi_info = {
      approx_kl: number;
      entropy: number;
      clip_frac: any;
    };
    const compute_loss_pi = (data: PPOBufferComputations): { loss_pi: tf.Tensor; pi_info: pi_info } => {
      const { obs, act, adv } = data;
      return tf.tidy(() => {
        const logp_old = data.logp.expandDims(-1);
        const adv_e = adv.expandDims(-1);
        const { pi, logp_a } = this.ac.pi.apply(obs, act);

        const ratio = logp_a!.sub(logp_old).exp();
        const clip_adv = ratio.clipByValue(1 - clip_ratio, 1 + clip_ratio).mul(adv_e);

        const adv_ratio = ratio.mul(adv_e);

        const ratio_and_clip_adv = tf.stack([adv_ratio, clip_adv]);

        const loss_pi = ratio_and_clip_adv.min(0).mean().mul(-1);

        const log_ratio = logp_a!.sub(logp_old);
        const approx_kl = log_ratio.exp().sub(1).sub(log_ratio).mean().arraySync() as number;

        const entropy = pi.entropy().mean().arraySync() as number;
        const clipped = ratio
          .greater(1 + clip_ratio)
          .logicalOr(ratio.less(1 - clip_ratio))
          .mean()
          .arraySync() as number;

        return {
          loss_pi,
          pi_info: {
            approx_kl,
            entropy,
            clip_frac: clipped,
          },
        };
      });
    };
    const compute_loss_vf = (data: PPOBufferComputations) => {
      const { obs, ret } = data;
      return tf.tidy(() => {
        const predict = this.ac.v.apply(obs).flatten();
        return predict.sub(ret).pow(2).mean();
      });
    };

    const update = async () => {
      const data = await buffer.get();
      const totalSize = configs.steps_per_iteration;
      const batchSize = configs.batch_size;

      let kls: number[] = [];
      let entropy = 0;
      let clip_frac = 0;

      let trained_pi_iters = 0;

      let continueTraining = true;

      for (let epoch = 0; epoch < configs.n_epochs; epoch++) {
        let batchStartIndex = 0;
        let batch = 0;
        let maxBatch = Math.floor(totalSize / batchSize);
        const indices = tf.tensor1d(Array.from(tf.util.createShuffledIndices(totalSize)), 'int32');
        while (batch < maxBatch) {
          const batchData = {
            obs: data.obs.gather(indices.slice(batchStartIndex, batchSize)),
            act: data.act.gather(indices.slice(batchStartIndex, batchSize)),
            adv: data.adv.gather(indices.slice(batchStartIndex, batchSize)),
            ret: data.ret.gather(indices.slice(batchStartIndex, batchSize)),
            logp: data.logp.gather(indices.slice(batchStartIndex, batchSize)),
          };

          // normalization adv
          const stats = {
            mean: batchData.adv.mean(),
            std: nj.std(batchData.adv.arraySync()),
          };
          batchData.adv = batchData.adv.sub(stats.mean).div(stats.std + 1e-8);

          batchStartIndex += batchSize;

          const grads = optimizer.computeGradients(() => {
            const { loss_pi, pi_info } = compute_loss_pi(batchData);
            kls.push(pi_info.approx_kl);
            entropy = pi_info.entropy;
            clip_frac = pi_info.clip_frac;

            const loss_v = compute_loss_vf(batchData);
            return loss_pi.add(loss_v.mul(configs.vf_coef)) as tf.Scalar;
          });
          if (kls[kls.length - 1] > 1.5 * target_kl) {
            log.warn(
              `${configs.name} | Early stopping at epoch ${epoch} batch ${batch}/${Math.floor(
                totalSize / batchSize
              )} of optimizing policy due to reaching max kl ${kls[kls.length - 1]} / ${1.5 * target_kl}`
            );
            continueTraining = false;
            break;
          }

          const maxNorm = 0.5;
          const clippedGrads: tf.NamedTensorMap = {};
          const totalNorm = tf.norm(tf.stack(Object.values(grads.grads).map((grad) => tf.norm(grad))));
          const clipCoeff = tf.minimum(tf.scalar(1.0), tf.scalar(maxNorm).div(totalNorm.add(1e-6)));
          Object.keys(grads.grads).forEach((name) => {
            clippedGrads[name] = tf.mul(grads.grads[name], clipCoeff);
          });

          optimizer.applyGradients(clippedGrads);
          batch++;
        }
        trained_pi_iters++;
        if (!continueTraining) {
          break;
        }
      }

      const metrics = {
        kl: nj.mean(nj.array(kls)),
        entropy,
        clip_frac,
        trained_pi_iters,
      };

      return metrics;
    };

    // const start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_len = 0;
    let ep_rets = [];
    for (let iteration = 0; iteration < configs.iterations; iteration++) {
      for (let t = 0; t < local_steps_per_epoch; t++) {
        let { a, v, logp_a } = this.ac.step(this.obsToTensor(o));
        const action = np.tensorLikeToNdArray(this.actionToTensor(a));
        const stepInfo = env.step(action);
        const next_o = stepInfo.observation;

        const r = stepInfo.reward;
        const d = stepInfo.done;
        ep_ret += r;
        ep_len += 1;

        if (env.actionSpace.meta.discrete) {
          a = a.reshape([-1, 1]);
        }

        buffer.store(
          np.tensorLikeToNdArray(this.obsToTensor(o)),
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
            log.warn(`${configs.name} | Trajectory cut off by epoch at ${ep_len} steps`);
          }
          let v = 0;
          if (timeout || epoch_ended) {
            v = (this.ac.step(this.obsToTensor(o)).v.arraySync() as number[][])[0][0];
          }
          buffer.finishPath(v, d);
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
        const msg = `${configs.name} | Iteration ${iteration} metrics: `;
        log.info(
          {
            ...metrics,
            ep_rets: ep_rets_metrics,
          },
          msg
        );
      }
      await configs.iterationCallback({
        iteration,
        ...metrics,
        ep_rets: ep_rets_metrics,
      });

      ep_rets = [];
    }
  }
}
