import { expect } from 'chai';
import { Spaces } from '../../src';

describe('Test discrete space', () => {
  describe('Test Discrete', () => {
    it('should sample correctly', () => {
      let space = new Spaces.Discrete(1);
      expect(space.contains(space.sample())).to.equal(true);
    });
    it('should say false for data not of appropriate type', () => {
      let space = new Spaces.Discrete(1);
      expect(space.contains(3)).to.equal(false);
      expect(space.contains(-1)).to.equal(false);
    });
  });
});
