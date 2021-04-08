import { expect } from 'chai';
import { Environments } from '../../src';

describe('Test Simple Grid World', () => {
  it('should run a episode properly', () => {
    let env = new Environments.Examples.SimpleGridWorld(4, 4, [{ x: 0, y: 3 }], { x: 0, y: 0 });
    let observation = env.reset();
    for (let step = 1; step < 4; step++) {
      let stepOut = env.step(2);
      let reward = stepOut.reward;
      let done = stepOut.done;
      observation = stepOut.observation;
      expect(reward).to.equal(-1);
      if (step < 3) {
        expect(done).to.equal(false);
      } else {
        expect(done).to.equal(true);
      }
      if (done) {
        env.reset();
      }
    }
  });
});
