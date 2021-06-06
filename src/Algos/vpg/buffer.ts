import * as tf from '@tensorflow/tfjs';
import nj, { NdArray } from 'numjs';
import * as np from 'rl-ts/lib/utils/np';
import * as core from 'rl-ts/lib/Algos/utils/core';
import { Tensor1D } from '@tensorflow/tfjs';
import * as ct from 'rl-ts/lib/utils/clusterTools';
export interface VPGBufferConfigs {
  obsDim: number[];
  actDim: number[];
  /** Buffer Size */
  size: number;
  gamma?: number;
  lam?: number;
}
export interface VPGBufferComputations {
  obs: tf.Tensor;
  act: tf.Tensor1D;
  ret: tf.Tensor1D;
  adv: tf.Tensor1D;
  logp: tf.Tensor1D;
}

/**
 * Buffer for VPG for storing trajectories experienced by a VPG agent.
 * Uses Generalized Advantage Estimation (GAE-Lambda) to calculate advantages of state-action pairs
 */
export class VPGBuffer {
  /** Observations buffer */
  public obsBuf: NdArray;
  /** Actions buffer */
  public actBuf: NdArray;
  /** Advantage estimates buffer */
  public advBuf: NdArray;
  /** Rewards buffer */
  public rewBuf: NdArray;
  /** Returns buffer */
  public retBuf: NdArray;
  /** Values buffer */
  public valBuf: NdArray;
  /** Log probabilities buffer */
  public logpBuf: NdArray;

  private ptr = 0;
  private pathStartIdx = 0;
  private maxSize = -1;

  public gamma = 0.99;
  public lam = 0.97;

  constructor(public configs: VPGBufferConfigs) {
    if (configs.gamma !== undefined) {
      this.gamma = configs.gamma;
    }
    if (configs.lam !== undefined) {
      this.lam = configs.lam;
    }

    this.obsBuf = nj.zeros([configs.size, ...configs.obsDim], 'float32');
    this.actBuf = nj.zeros([configs.size, ...configs.actDim], 'float32');
    this.advBuf = nj.zeros([configs.size], 'float32');
    this.rewBuf = nj.zeros([configs.size], 'float32');
    this.retBuf = nj.zeros([configs.size], 'float32');
    this.valBuf = nj.zeros([configs.size], 'float32');
    this.logpBuf = nj.zeros([configs.size], 'float32');
    this.maxSize = configs.size;
  }
  public store(obs: NdArray, act: NdArray, rew: number, val: number, logp: number) {
    if (this.ptr >= this.maxSize) throw new Error('Experience Buffer has no room');
    const slice = [this.ptr, this.ptr + 1];
    this.obsBuf.slice(slice).assign(obs, false);
    this.actBuf.slice(slice).assign(act, false);
    this.rewBuf.set(this.ptr, rew);
    this.valBuf.set(this.ptr, val);
    this.logpBuf.set(this.ptr, logp);
    this.ptr += 1;
  }

  public finishPath(lastVal = 0) {
    const path_slice = [this.pathStartIdx, this.ptr];
    const rews = np.push(this.rewBuf.slice(path_slice), lastVal);
    const vals = np.push(this.valBuf.slice(path_slice), lastVal);
    // GAE Lambda Advantage = sum (gamma lambda)^h delta_{t+h, 0}
    // compute delta_{t+h, 0}
    // replicates deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    const deltas = rews
      .slice([0, -1])
      .add(vals.slice(1).multiply(this.gamma))
      .subtract(vals.slice([0, -1]));

    // compute GAE-Lambda advantage, assign in place.
    this.advBuf.slice(path_slice).assign(core.discountCumSum(deltas, this.gamma * this.lam), false);

    // compute rewards-to-go
    this.retBuf.slice(path_slice).assign(core.discountCumSum(rews, this.gamma).slice([0, -1]), false);

    this.pathStartIdx = this.ptr;
  }

  public async get(): Promise<VPGBufferComputations> {
    if (this.ptr !== this.maxSize) {
      throw new Error("Buffer isn't full yet!");
    }
    this.pathStartIdx = 0;
    this.ptr = 0;

    // move to tensors for use by update method and nicer functions
    let advBuf = np.toTensor(this.advBuf);

    // normalization trick
    const stats = await ct.statisticsScalar(advBuf, { max: true, min: true }, true);
    advBuf = advBuf.sub(stats.mean).div(stats.std);
    this.advBuf = await np.fromTensor(advBuf);
    return {
      obs: np.toTensor(this.obsBuf),
      act: np.toTensor(this.actBuf) as Tensor1D,
      ret: np.toTensor(this.retBuf) as Tensor1D,
      adv: advBuf as Tensor1D,
      logp: np.toTensor(this.logpBuf) as Tensor1D,
    };
  }
}
