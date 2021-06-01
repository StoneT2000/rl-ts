import * as tf from '@tensorflow/tfjs';
import * as np from 'rl-ts/lib/utils/np';
import { NdArray } from 'numjs';
import { Distribution } from '.';
const logsqrtpi2 = Math.log(Math.sqrt(Math.PI * 2));
/**
 * A normal distribution
 */
export class Normal extends Distribution {
  public mean: NdArray;
  public std: NdArray;

  constructor(public tf_mean: tf.Tensor, public tf_std: tf.Tensor) {
    super(tf_mean.shape, 'Normal');
    if (!np.arrayEqual(tf_mean.shape, tf_std.shape)) {
      throw new Error(`mean and std have different shapes - ${tf_mean.shape} and ${tf_std.shape}`);
    }
    this.mean = np.tensorLikeToNdArray(tf_mean);
    this.std = np.tensorLikeToNdArray(tf_std);
  }
  sample(): tf.Tensor {
    const sample = tf.buffer(this.mean.shape, 'float32');
    for (let i = 0; i < sample.size; i++) {
      const loc = sample.indexToLoc(i);
      const value = tf.randomNormal([1], this.mean.get(...loc), this.std.get(...loc));
      sample.set(value.dataSync()[0], ...loc);
    }
    return sample.toTensor();
  }
  logProb(value: tf.Tensor): tf.Tensor {
    // TODO: consider using cwise ourself to implement a nj logarithm function (which for some reason is missing!) or just loop through
    // const _variance = tf.buffer(this.mean.shape);
    // const _logScale = tf.buffer(this.mean.shape);
    // for (let i = 0; i < _variance.size; i++) {
    //   let loc = _variance.indexToLoc(i);
    //   let std = this.std.get(...loc)
    //   _variance.set(std ** 2, ...loc);
    //   _logScale.set(Math.log(std), ...loc);
    // }
    // this.tf_std

    const variance = this.tf_std.pow(2);
    const logScale = tf.log(this.tf_std);
    // const logScale = _logScale.toTensor();
    // console.log(variance.print())
    // variance.slice([0, 1], [0, 1]).print();
    const denom = variance.mul(2);
    return value.sub(this.tf_mean).pow(2).neg().div(denom).sub(logScale).sub(logsqrtpi2);
  }
  entropy() {
    const _logScale = tf.buffer(this.mean.shape);
    for (let i = 0; i < _logScale.size; i++) {
      const loc = _logScale.indexToLoc(i);
      const std = this.std.get(...loc);
      _logScale.set(Math.log(std), ...loc);
    }
    const logScale = _logScale.toTensor();
    return logScale.add(0.5 + 0.5 * Math.log(2 * Math.PI));
  }
}
