import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs';
import { gatherOriginal, gatherOwn } from '../../src/utils/gather';
describe('gather', () => {
  it('it should work correctly', () => {
    const input = tf.tensor2d([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    const indices = tf.tensor1d([0, 2, 1], 'int32');
    expect(gatherOriginal(input, indices).arraySync()).to.eql([1, 7, 10]);
    expect(gatherOwn(input, indices).arraySync()).to.eql([1, 7, 10]);
    const input2 = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices2 = tf.tensor1d([0, 1, 1], 'int32');
    expect(gatherOriginal(input2, indices2).arraySync()).to.eql([1, 4, 6]);
    expect(gatherOwn(input2, indices2).arraySync()).to.eql([1, 4, 6]);
  });
});
