import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs';
import { Normal } from '../../../src/utils/Distributions/normal';
describe('Test normal distribution class', () => {
  it('should sample correctly', () => {
    const mean = tf.range(1, 5, 1, 'float32');
    const std = tf.tensor([1e-1, 1e-1, 1e-1, 1e-1]);
    const normal = new Normal(mean, std);
    const sample = normal.sample();
    const data = sample.dataSync();
    for (let i = 0; i < sample.size; i++) {
      expect(data[i]).to.be.lessThan(i + 1 + 1e-1 * 3);
      expect(data[i]).to.be.greaterThan(i + 1 - 1e-1 * 3);
    }
  });
  it('should compute log prob correctly', () => {
    const mean = tf.range(1, 5, 1, 'float32');
    const std = tf.tensor([1e-1, 1e-1, 1e-1, 1e-1]);
    const normal = new Normal(mean, std);
    const values = tf.tensor([1, 2 - 1e-1, 3 + 1e-1, 4 + 2e-2]);
    const logProbs = normal.logProb(values).dataSync();
    const expectedRes = [1.38364656, 0.8836463689804077, 0.8836463689804077, 1.36364656];
    for (let i = 0; i < logProbs.length; i++) {
      // the computation varies is inconsistently low res
      expect(logProbs[i]).to.be.closeTo(expectedRes[i], 1e-5);
    }
  });
});
