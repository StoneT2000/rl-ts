/**
 * Tensorflow js based distributions because tensorflow probability (tfp) has not been ported over yet :(
 */
import * as tf from '@tensorflow/tfjs';

export abstract class Distribution {
  constructor(public shape: number[], public name: string) {}
  abstract sample(): tf.Tensor;
  abstract logProb(value: tf.Tensor): tf.Tensor;
  abstract entropy(): tf.Tensor;
}
