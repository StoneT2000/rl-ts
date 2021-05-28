import { expect } from 'chai';
import { Spaces } from '../../src/';
import * as np from '../../src/RL/utils/np';
describe('Test box space', () => {
  describe('Test PrimitiveBox2D', () => {
    it('should sample correctly', () => {
      const space = new Spaces.PrimitiveBox2D(0, 1, [2, 2]);
      expect(space.contains(space.sample())).to.equal(true);
    });
    it('should say false for data not of appropriate type', () => {
      const space = new Spaces.PrimitiveBox2D(0, 1, [2, 2]);
      const sample = space.sample();
      sample[0][0] = 23.4;
      expect(space.contains(sample)).to.equal(false);
      expect(space.contains([[2, 3]])).to.equal(false);
    });
  });
  describe('Test Box', () => {
    it('should sample correctly with scalar low and high', () => {
      const space = new Spaces.Box(0, 1, [2, 2], 'float32');
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.lt(1);
        expect(sample.selection.data[i]).to.be.gt(0);
      }
    });
    it('should sample correctly with ndarray low and high', () => {
      const space = new Spaces.Box(
        np.pack([
          [0, 1],
          [2, 3],
        ]),
        np.pack([
          [1, 2],
          [3, 4],
        ]),
        [2, 2],
        'float32'
      );
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.lt(i + 1);
        expect(sample.selection.data[i]).to.be.gt(i);
      }
    });
    it('should sample correctly with ndarray low and scalar high', () => {
      const space = new Spaces.Box(
        np.pack([
          [0, 1],
          [2, 3],
        ]),
        10,
        [2, 2],
        'float32'
      );
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.lt(10);
        expect(sample.selection.data[i]).to.be.gt(i);
      }
    });
    it('should sample correctly with scalar low and ndarray high', () => {
      const space = new Spaces.Box(
        -10,
        np.pack([
          [0, 1],
          [2, 3],
        ]),
        [2, 2],
        'float32'
      );
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.lt(i + 1);
        expect(sample.selection.data[i]).to.be.gt(-10);
      }
    });
    it('should sample correctly with low only', () => {
      const space = new Spaces.Box(-10, null, [2, 2], 'float32');
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.gt(-10);
      }
    });
    it('should sample correctly with high only', () => {
      const space = new Spaces.Box(null, 10, [2, 2], 'float32');
      const sample = space.sample();
      for (let i = 0; i < sample.size; i++) {
        expect(sample.selection.data[i]).to.be.lt(10);
      }
    });
  });
});
