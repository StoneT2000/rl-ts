import { expect } from 'chai';
import { discountCumSum } from '../../../src/RL/Algos/utils/core';
import * as np from '../../../src/RL/utils/np';
describe('Test Algos utils', () => {
  it('discountCumSum should work', () => {
    const x = np.pack([1, 2, 3, 1, 2, 3]);
    const discount = 0.25;
    const res = np.unpack(discountCumSum(x, discount));
    const expectedRes = [1.71386719, 2.85546875, 3.421875, 1.6875, 2.75, 3];
    for (let i = 0; i < expectedRes.length; i++) {
      expect(res[i]).to.be.closeTo(expectedRes[i], 1e-10);
    }
  });
});
