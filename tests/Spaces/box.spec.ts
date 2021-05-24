import { expect } from 'chai';
import { Spaces } from '../../src';

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
});
