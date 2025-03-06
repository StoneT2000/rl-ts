import { expect } from 'chai';
import { Environments } from '../../src';

describe('Test Simple Grid World', () => {
  it('should run a episode properly', async () => {
    const env = new Environments.Examples.SimpleGridWorld(4, 4, [{ x: 0, y: 3 }], { x: 0, y: 0 });
    // eslint-disable-next-line
    const observation = await env.reset();
    for (let step = 1; step < 4; step++) {
      const stepOut = await env.step(2);
      const reward = stepOut.reward;
      const done = stepOut.done;
      expect(reward).to.equal(-1);
      if (step < 3) {
        expect(done).to.equal(false);
      } else {
        expect(done).to.equal(true);
      }
      if (done) {
        await env.reset();
      }
    }
  });
});
