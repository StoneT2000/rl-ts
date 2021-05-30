/**
 * Various tooling for computing values across Node.js processes to take advantage of multi-core systems via the cluster library
 * 
 * Primary / Master process aggregates messages received from other workers
 */
import * as tf from '@tensorflow/tfjs-node';
import cluster from 'cluster';
import * as mpi from './mpi';
import { OPS } from './ops';

/**
 * Compute various statistics of a registered variable
 */
export const statisticsScalar = () => {

}

/**
 * 
 * @param x - tensor with shape [...shape]
 * @param rest - tensor with shape [P, ...shape]
 */
const handleOp = async(x: tf.Tensor, rest: tf.Tensor, op: OPS) => {
  switch(op) {
    case OPS.SUM:
      return x.add(rest.sum(0))
    case OPS.GATHER:
      return x.concat(rest);
  }
}

/**
 * Reduces input elements in each process according to op given and returns the output to the root to then perform op again
 * @param x 
 * @param op 
 */
export const reduce = async (x: tf.Tensor, op: OPS) => {
  if (cluster.isMaster) {
    let results: tf.Tensor[] = [];
    let res = await mpi.receiveAll();
    let val = tf.tensor(res.map((v) => v.data! as number)); // [P, ...shape] (P = number of workers)
    switch(op) {
      case OPS.SUM:
        return x.add(val.sum(0))
    }
  } else  if (cluster.isWorker) {
    await mpi.send({type: mpi.MessageType.SEND, data: mpi.tensorToSerializable(x)}, process);
  }
}

/**
 * Calls reduce and then the primary broadcasts result to all workers
 * @param x 
 * @param op 
 */
export const allreduce = async (x: tf.Tensor, op: OPS) => {
  if (cluster.isMaster) {
    let results: tf.Tensor[] = [];
    let res = await mpi.receiveAll();
    let val = tf.tensor(res.map((v) => v.data! as number)); // [P, ...shape] (P = number of workers)
    val = await handleOp(x, val, op);
    await mpi.sendall({type: mpi.MessageType.SEND, data: mpi.tensorToSerializable(val)})
    return val;
  } else {
    // workers wait for a message from master
    await mpi.send({type: mpi.MessageType.SEND, data: mpi.tensorToSerializable(x)}, process);
    return tf.tensor((await mpi.receive()).data! as number[]);
  }
}

export const sum = async (x: tf.Tensor) => {
  return allreduce(x, OPS.SUM);
}

export const avg = async (x: tf.Tensor) => {
  let val = await allreduce(x, OPS.SUM);
  return val.div(mpi.numProcs());
}

export * from './mpi';