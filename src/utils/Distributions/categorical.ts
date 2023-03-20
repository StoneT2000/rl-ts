import * as tf from '@tensorflow/tfjs';
import nj, { NdArray } from 'numjs';
import { tensorLikeToNdArray, tensorLikeToTensor } from 'rl-ts/lib/utils/np';
import { Distribution } from '.';
import { gatherOwn } from '../gather';

/**
 * A categorical distribution
 */
export class Categorical extends Distribution {
  public logits: NdArray;
  public tf_logits: tf.Tensor;

  constructor(init_tf_logits: tf.Tensor) {
    super(init_tf_logits.shape, 'Categorical');
    const tf_logits = init_tf_logits.sub(init_tf_logits.logSumExp(-1, true));
    this.logits = tensorLikeToNdArray(tf_logits);
    this.tf_logits = tf_logits;
  }
  sample(): tf.Tensor {
    let logits = this.logits_parameter();
    const sample = tf.buffer([logits.shape[0]], 'float32');
    const logits_2d = tf.reshape(logits, [-1, this._num_categories(logits)]);
    for (let i = 0; i < sample.size; i++) {
      const loc = sample.indexToLoc(i);
      const value = tf.multinomial(logits_2d as tf.Tensor2D, 1);
      sample.set(value.dataSync()[0], ...loc);
    }
    return sample.toTensor();
  }

  private _num_categories(logits: tf.Tensor): number {
    return logits.shape[logits.shape.length - 1];
  }

  logProb(value: tf.Tensor): tf.Tensor {
    value = tf.cast(value, 'int32');
    const logProb = gatherOwn(this.tf_logits, value);
    return logProb;
  }
  logits_parameter() {
    return tensorLikeToTensor(this.logits);
  }
  entropy() {
    const probs = tf.softmax(this.logits_parameter());
    const p_log_p = tf.mul(this.logits_parameter(), probs);
    return tf.neg(p_log_p.sum(-1));
  }
}
