import { expect } from 'chai';
import ndarray from 'ndarray';
import * as np from '../../src/RL/utils/np';
describe('Test numpy (in ts) utils', () => {
  it('should set correctly', () => {
    const a = np.zeros([3, 2, 2]);
    const b = np.pack([
      [20, 3],
      [4, 2],
    ]);
    const c = np.set(a, [0], b);
    // in place
    expect(c.data).to.eql(a.data, 'should replace in-place');
    const res = [20, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    expect(np.arrayEqual(c.data as number[], res)).to.equal(true, 'should set correctly');

    const d = np.set(a, [2, 1, 1], ndarray([1], [1]));
    res[res.length - 1] = 1;
    expect(np.arrayEqual(d.data as number[], res)).to.equal(true, 'should set correctly');
  });
  it('should create a matrix of zeros correctly', () => {
    const a = np.zeros([3, 4]);
    for (let i = 0; i < a.data.length; i++) {
      expect(a.data[i]).to.equal(0);
    }
  });
  it('should push correctly', () => {
    const a = np.pack([2,3,4,5]);
    const b = np.push(a, 4);
    expect(np.arrayEqual(b.data as number[], [2,3,4,5,4])).to.equal(true);
    expect(np.arrayEqual(a.data as number[], [2,3,4,5])).to.equal(true);
  });
});
