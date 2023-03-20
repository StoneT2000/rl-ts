import * as tf from '@tensorflow/tfjs';

export function gatherOriginal(input: tf.Tensor, indices: tf.Tensor): tf.Tensor {
  return tf.gather(input, indices, 1, 1);
}

// bug in tf.gather during gradient computation
export function gatherOwn(input: tf.Tensor, indices: tf.Tensor): tf.Tensor {
  const batchSize = input.shape[0];
  const tensors = input.split(batchSize, 0);
  const output = tf.stack(tensors.map((tensor, i) => tensor.squeeze().gather((indices.arraySync() as number[])[i])));
  return output;
}
