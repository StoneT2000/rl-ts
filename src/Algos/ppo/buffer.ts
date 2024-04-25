import * as tf from '@tensorflow/tfjs';
import nj, { NdArray } from 'numjs';
import * as np from 'rl-ts/lib/utils/np';
import { Tensor1D } from '@tensorflow/tfjs';
export interface PPOBufferConfigs {
  obsDim: number[];
  actDim: number[];
  /** Buffer Size */
  size: number;
  gamma?: number;
  lam?: number;
}
export interface PPOBufferComputations {
  obs: tf.Tensor;
  act: tf.Tensor1D;
  ret: tf.Tensor1D;
  adv: tf.Tensor1D;
  logp: tf.Tensor1D;
}

/**
 * Buffer for PPO for storing trajectories experienced by a PPO agent.
 * Uses Generalized Advantage Estimation (GAE-Lambda) to calculate advantages of state-action pairs
 */
export class PPOBuffer {
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

  constructor(public configs: PPOBufferConfigs) {
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

  public finishPath(lastVal = 0, done: boolean) {
    const path_slice = [this.pathStartIdx, this.ptr];
    let lastGaeLam = 0;
    for (let t = this.ptr - 1; t >= this.pathStartIdx; t--) {
      const nextVal = t === this.ptr - 1 ? lastVal : this.valBuf.get(t + 1);
      const nextNonTerminal = t === this.ptr - 1 ? (done ? 0 : 1) : 1;
      const delta = this.rewBuf.get(t) + this.gamma * nextVal * nextNonTerminal - this.valBuf.get(t);
      lastGaeLam = delta + this.gamma * this.lam * nextNonTerminal * lastGaeLam;
      this.advBuf.set(t, lastGaeLam);
    }
    this.retBuf.slice(path_slice).assign(this.advBuf.slice(path_slice).add(this.valBuf.slice(path_slice)), false);

    this.pathStartIdx = this.ptr;
  }

  public async get(): Promise<PPOBufferComputations> {
    if (this.ptr !== this.maxSize) {
      throw new Error("Buffer isn't full yet!");
    }
    this.pathStartIdx = 0;
    this.ptr = 0;

    return {
      obs: np.toTensor(this.obsBuf),
      act: np.toTensor(this.actBuf) as Tensor1D,
      ret: np.toTensor(this.retBuf) as Tensor1D,
      adv: np.toTensor(this.advBuf) as Tensor1D,
      logp: np.toTensor(this.logpBuf) as Tensor1D,
    };
  }
}
