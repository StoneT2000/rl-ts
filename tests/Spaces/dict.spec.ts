import { expect } from 'chai';
import { PrimitiveBox2D, Dict, Discrete } from '../../lib/es6/RL/Spaces';

describe('Test dict space', () => {
  it('should sample correctly from one nesting', () => {
    const space = new Dict({
      box: new PrimitiveBox2D(0, 1, [1, 3]),
      val: new Discrete(4),
    });
    expect(space.contains(space.sample())).to.equal(true);
  });
  it('should say false for data not of appropriate type', () => {
    const space = new Dict({
      box: new PrimitiveBox2D(0, 1, [1, 3]),
      val: new Discrete(4),
    });
    let sample = space.sample();
    sample.val = 5;
    expect(space.contains(sample)).to.equal(false);
    sample = space.sample();
    sample.box[0][2] = 5;
    expect(space.contains(sample)).to.equal(false);
  });
  it('should sample correctly from two nestings', () => {
    const space = new Dict({
      box: new PrimitiveBox2D(0, 1, [1, 3]),
      val: new Discrete(4),
      anotherDict: new Dict({
        anotherVal: new Discrete(20),
      }),
    });
    expect(space.contains(space.sample())).to.equal(true);
  });
  it('should say false for data not of appropriate type with two nestings', () => {
    const space = new Dict({
      box: new PrimitiveBox2D(0, 1, [1, 3]),
      val: new Discrete(4),
      anotherDict: new Dict({
        anotherVal: new Discrete(20),
      }),
    });
    const sample = space.sample();
    sample.anotherDict.anotherVal = -2;
    expect(space.contains(sample)).to.equal(false);
  });
});
