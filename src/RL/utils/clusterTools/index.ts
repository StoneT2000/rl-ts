/**
 * Various tooling for computing values across Node.js processes to take advantage of multi-core systems via the cluster library
 *
 * Primary / Master process aggregates messages received from other workers
 */
import * as tf from '@tensorflow/tfjs-node';
import { NamedTensorMap } from '@tensorflow/tfjs-node';
import cluster from 'cluster';
import * as mpi from './mpi';
import { OPS } from './ops';

/**
 * Compute various statistics of a registered variable
 */
export const statisticsScalar = () => {};

/**
 *
 * @param x - tensor with shape [...shape]
 * @param rest - tensor with shape [P, ...shape]
 */
const handleOp = async (x: tf.Tensor, rest: tf.Tensor, op: OPS) => {
  switch (op) {
    case OPS.SUM:
      return x.add(rest.sum(0));
    case OPS.GATHER:
      return x.concat(rest);
  }
};

/**
 *
 * @param x - tensor with shape [...shape]
 * @param rest - tensor with shape [P, ...shape]
 */
 const handleOpNumber = async (x: number, rest: number[], op: OPS) => {
  switch (op) {
    case OPS.SUM: {
      let val = 0;
      for (const v of rest) {
        val += v;
      }
      return x + val;
    }
    default:
      throw new Error(`${op} is invalid op`)
  }
};

/**
 * Reduces input elements in each process according to op given and returns the output to the root to then perform op again
 * @param x
 * @param op
 */
export const reduce = async (x: tf.Tensor, op: OPS) => {
  if (mpi.numProcs() === 1) return x;
  if (cluster.isMaster) {
    const res = await mpi.receiveAll();
    const val = tf.tensor(res.map((v) => v.data! as number)); // [P, ...shape] (P = number of workers)
    switch (op) {
      case OPS.SUM:
        return x.add(val.sum(0));
    }
  } else if (cluster.isWorker) {
    await mpi.send({ type: mpi.MessageType.SEND , data: mpi.tensorToSerializable(x) }, process);
  }
};

/**
 * Performs reduction and then broadcasts result to all workers
 * @param x
 * @param op
 */
export const allreduce = async (x: tf.Tensor, op: OPS) => {
  if (mpi.numProcs() === 1) return x;
  // TODO: allow this to wait a specific message and not a generic one by a specified key. This will make ops like averageGradients faster.
  if (cluster.isMaster) {
    const res = await mpi.receiveAll();
    let val = tf.tensor(res.map((v) => v.data! as number)); // [P, ...shape] (P = number of workers)
    val = await handleOp(x, val, op);
    await mpi.sendall({ type: mpi.MessageType.SEND, data: mpi.tensorToSerializable(val) });
    return val;
  } else {
    // workers wait for a message from master
    await mpi.send({ type: mpi.MessageType.SEND , data: mpi.tensorToSerializable(x) }, process);
    return tf.tensor((await mpi.receive()).data! as number[]);
  }
};

/**
 * Performs reduction and then broadcasts result to all workers
 * @param x
 * @param op
 */
 export const allreduceNumber = async (x: number, op: OPS) => {
  if (mpi.numProcs() === 1) return x;
  if (cluster.isMaster) {
    const res = await mpi.receiveAll();
    const val = res.map((v) => v.data! as number); // [P, ...shape] (P = number of workers)
    const ret = await handleOpNumber(x, val, op);
    await mpi.sendall({ type: mpi.MessageType.SEND , data: ret });
    return ret;
  } else {
    // workers wait for a message from master
    await mpi.send({ type: mpi.MessageType.SEND, data: x }, process);
    return (await mpi.receive()).data as number;
  }
};

export const sum = async (x: tf.Tensor) => {
  if (mpi.numProcs() === 1) return x;
  return allreduce(x, OPS.SUM);
};

export const avg = async (x: tf.Tensor) => {
  if (mpi.numProcs() === 1) return x;
  const val = await sum(x);
  return val.div(mpi.numProcs());
};

export const sumNumber = async (x: number) => {
  if (mpi.numProcs() === 1) return x;
  return allreduceNumber(x, OPS.SUM);
}

export const avgNumber = async(x: number) => {
  if (mpi.numProcs() === 1) return x;
  const val = await sumNumber(x);
  return val / mpi.numProcs();
}

/** averages gradients produced by optimizer.computeGradients function in tensorflow. Stores averaged values in place */
export const averageGradients = async (grads: NamedTensorMap) => {
  if (mpi.numProcs() === 1) return grads;
  for (const key of Object.keys(grads)) {
    grads[key] = await avg(grads[key])
  }
  return grads;
};

export * from './mpi';
