import * as tf from '@tensorflow/tfjs-node';
import { OPS } from './ops';
import cluster from 'cluster';
import { Serializable } from 'node:child_process';
import os from 'os';

export enum MessageType {
  INIT,
  SEND,
}
export interface Message {
  type: MessageType;
  data?: Serializable;
}

// let receivePromise: Promise<Message>;
const numCPUs = os.cpus().length;
let workers: cluster.Worker[] = [];
let _procCount = 1;
let _id = 0;
export const setupMPI = async () => {
  // receivePromise = new Promise((resolve, reject) => {
  // process.once('message', (m) => {
  //   resolve(m);
  // });
  // });
};

export const fork = async (forkCount: number) => {
  const listeningPromises: Array<Promise<cluster.Worker>> = [];
  if (cluster.isMaster) {
    for (let i = 0; i < forkCount; i++) {
      const worker = cluster.fork();
      listeningPromises.push(
        new Promise((res, rej) => {
          worker.on('online', async () => {
            await send({ type: MessageType.INIT, data: [worker.id, forkCount + 1] }, worker);
            res(worker);
          });
          worker.on('error', (err) => {
            rej(err);
          });
        })
      );
    }
    workers = await Promise.all(listeningPromises);
    _procCount = forkCount + 1;
    _id = 0;
  } else {
    // wait for message
    let data = (await receive()).data! as number[];
    _id = data[0];
    _procCount = data[1];
  }
};
export const send = async (msg: Message, worker: cluster.Worker | NodeJS.Process): Promise<void> => {
  return new Promise((resolve, reject) => {
    worker.send!(msg, undefined, (err) => {
      if (err) reject(err);
      resolve();
    });
  });
};
export const sendall = async (msg: Message) => {
  let results: Promise<void>[] = [];
  workers.forEach((worker) => {
    results.push(send(msg, worker));
  });
  return Promise.all(results);
};

export const receiveFromWorker = async (worker: cluster.Worker): Promise<Message> => {
  return new Promise((resolve, reject) => {
    worker.once('message', (m) => {
      resolve(m);
    });
  });
};

export const receiveAll = async () => {
  let results: Promise<Message>[] = [];
  workers.forEach((worker) => {
    results.push(receiveFromWorker(worker));
  });
  return Promise.all(results);
};

export const receive = async (): Promise<Message> => {
  return new Promise((resolve, reject) => {
    process.once('message', (m) => {
      resolve(m);
    });
  });
};

// disconnects all workers and stops them
export const disconnect = async () => {
  workers.forEach((worker) => {
    worker.disconnect();
  });
};

export const tensorToSerializable = (x: tf.Tensor): Serializable => {
  return x.arraySync();
};

export const numProcs = () => {
  return _procCount;
};
export const id = () => {
  return _id;
};
