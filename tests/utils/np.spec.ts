import { expect } from 'chai';
import * as np from '../../src/RL/utils/np';
import nj from 'numjs';
describe('Test numpy (in ts) utils', () => {
  it('should set correctly', () => {
    const a = np.zeros([3, 2, 2]);
    const b = np.pack([
      [20, 3],
      [4, 2],
    ]);
    const c = np.set(a, [0], np.fromNj(b));
    // in place
    expect(c.data).to.eql(a.data, 'should replace in-place');
    const res = [20, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    expect(np.arrayEqual(c.data as number[], res)).to.equal(true, 'should set correctly');

    const d = np.set(a, [2, 1, 1], nj.array([1]));
    res[res.length - 1] = 1;
    expect(np.arrayEqual(d.data as number[], res)).to.equal(true, 'should set correctly');
  });
  it('should set by boolean mask correctly', () => {
    const mask = np.packNp([
      [1, 1],
      [0, 0],
    ]);
    const A = np.packNp([
      [20, 30],
      [40, 50],
    ]);
    const val = np.packNp([2, 3]);
    const res = np.set(A, mask, val);
    expect(np.arrayEqual(res.data as number[], [2, 3, 40, 50])).to.equal(true);

    const mask2 = np.packNp([
      [1, 1],
      [1, 0],
    ]);
    const B = np.packNp([
      [20, 30],
      [40, 50],
    ]);
    const val2 = np.packNp([2, 3, 4]);
    const res2 = np.set(B, mask2, val2);
    expect(np.arrayEqual(res2.data as number[], [2, 3, 4, 50])).to.equal(true);

    const mask3 = np.packNp([
      [0, 0],
      [0, 0],
    ]);
    const C = np.packNp([
      [20, 30],
      [40, 50],
    ]);
    const val3 = np.packNp([]);
    const res3 = np.set(C, mask3, val3);
    expect(np.arrayEqual(res3.data as number[], [20, 30, 40, 50])).to.equal(true);
  });
  it('should create a matrix of zeros correctly', () => {
    const a = np.zeros([3, 4]);
    for (let i = 0; i < a.data.length; i++) {
      expect(a.data[i]).to.equal(0);
    }
  });
  it('should push correctly', () => {
    const a = np.pack([2, 3, 4, 5]);
    const b = np.push(a, 4);
    expect(np.arrayEqual(b.selection.data as number[], [2, 3, 4, 5, 4])).to.equal(true);
    expect(np.arrayEqual(a.selection.data as number[], [2, 3, 4, 5])).to.equal(true);
  });
  it('should unsqueeze correctly', () => {
    let x = nj.zeros([3, 3, 3]);
    let ref = x.reshape(1, ...x.shape);
    let a = np.unsqueeze(x, 0);
    expect(np.arrayEqual(a.selection.stride, ref.selection.stride)).to.equal(true);
    expect(np.arrayEqual(a.selection.shape, ref.selection.shape)).to.equal(true);
    ref = ref.reshape(3, 1, 3, 3);
    np.unsqueeze(x, 1, false);
    expect(np.arrayEqual(x.selection.stride, ref.selection.stride)).to.equal(true);
    expect(np.arrayEqual(x.selection.shape, ref.selection.shape)).to.equal(true);
  });
});
