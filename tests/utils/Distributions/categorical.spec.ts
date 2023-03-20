import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs';
import { Categorical } from '../../../src/utils/Distributions/categorical';
import { tensorLikeToTensor } from '../../../src/utils/np';
describe('Test categorical distribution class', () => {
  it('should sample correctly', () => {
    const logits = tf.tensor([
      [-0.1811, 0.1901],
      [-0.1671, 0.1757],
      [0.2334, -0.2444],
      [0.0479, -0.0503],
    ]);
    const categorical = new Categorical(logits);
    const sample = categorical.sample();
    expect(sample.shape).to.eql([4]);
    for (let i = 0; i < sample.size; i++) {
      expect([0, 1]).to.include(sample.dataSync()[i]);
    }
  });

  it('should compute log prob correctly', () => {
    const logits = tf.tensor([
      [-0.1811, 0.1901],
      [-0.1671, 0.1757],
      [0.2334, -0.2444],
      [0.0479, -0.0503],
    ]);

    const expNormalizedLogits = [
      [-0.8959, -0.5247],
      [-0.8792, -0.5363],
      [-0.4825, -0.9603],
      [-0.6453, -0.7434],
    ];

    const categorical = new Categorical(logits);
    const normalizedLogits = tensorLikeToTensor(categorical.logits).arraySync() as number[][];
    for (let i = 0; i < normalizedLogits.length; i++) {
      const element = normalizedLogits[i];
      expect(element[0]).to.be.closeTo(expNormalizedLogits[i][0], 1e-4);
      expect(element[1]).to.be.closeTo(expNormalizedLogits[i][1], 1e-4);
    }
    const values = tf.tensor([0, 0, 1, 1]);
    const logProbs = categorical.logProb(values).arraySync() as number[];
    const expectedRes = [-0.8959, -0.8792, -0.9603, -0.7434];
    for (let i = 0; i < logProbs.length; i++) {
      // the computation varies is inconsistently low res
      expect(logProbs[i]).to.be.closeTo(expectedRes[i], 1e-4);
    }
  });

  it('should compute entropy correctly', () => {
    const logits = tf.tensor([
      [-0.0726, 0.0754],
      [0.1753, -0.1817],
      [0.1862, -0.193],
      [0.034, -0.0354],
    ]);

    const expEntropy = [0.6904, 0.6775, 0.6755, 0.6925];

    const categorical = new Categorical(logits);
    const entropy = categorical.entropy().dataSync();
    for (let i = 0; i < entropy.length; i++) {
      // the computation varies is inconsistently low res
      expect(entropy[i]).to.be.closeTo(expEntropy[i], 1e-4);
    }
  });
});
