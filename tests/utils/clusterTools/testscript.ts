import * as tf from '@tensorflow/tfjs-node';
import * as ct from '../../../src/RL/utils/clusterTools';
import { strict as assert } from 'assert';
import { sleep } from '../../../src/RL/utils/sleep';

const main = async () => {
  await ct.fork(2);

  // test synchronization
  if (ct.id() === 1) {
    await sleep(1000);
  }
  let val = await ct.sumNumber(10);
  assert.equal(val, 30);
  // test synchronization
  if (ct.id() === 0) {
    await sleep(100);
  }
  val = await ct.sumNumber(15);
  assert.equal(val, 45);


  // each process sends input to primary to then sum up element wise.
  const id = ct.id();
  if (id === 0) {
    console.info('Testing numProcs');
    assert.equal(ct.numProcs(), 3, 'numProcs incorrect');
  }

  // Test averaging
  const res = await ct.avg(
    tf
      .tensor([
        [1, 2, 3, 4],
        [-1.5, -0.5, -3.5, -4],
      ])
      .add(id)
  );
  if (id === 0) {
    console.info('Testing average');
    const data = res.arraySync();
    assert.deepEqual(
      data,
      [
        [2, 3, 4, 5],
        [-0.5, 0.5, -2.5, -3],
      ],
      'Incorrect average computation'
    );
  }

  try {
    await ct.disconnect();
  } catch (err) {
    assert.fail('Disconnect failed');
  }
};

main();
