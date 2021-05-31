import * as tf from '@tensorflow/tfjs-node';
import cluster from 'cluster';
import { Serializable } from 'node:child_process';

export enum MessageType {
  INIT,
  SEND,
}
export interface Message {
  type: MessageType;
  data?: Serializable;
}

let workers: cluster.Worker[] = [];
let _procCount = 1;
let _id = 0;
const messageBuffer: Message[] = [];
// export const setupMPI = async () => {
// TODO: do any setup here if needed
// };

export const fork = async (forkCount: number) => {
  const listeningPromises: Array<Promise<cluster.Worker>> = [];
  if (cluster.isMaster) {
    for (let i = 0; i < forkCount; i++) {
      const worker = cluster.fork();
      listeningPromises.push(
        new Promise((res, rej) => {
          worker.on('online', async () => {
            res(worker);
          });
          worker.on('message', (m) => {
            messageBuffer.push(m);
          });
          worker.on('error', (err) => {
            rej(err);
          });
        })
      );
    }
    workers = await Promise.all(listeningPromises);
    const workerInitSignals: Promise<void>[] = [];
    for (const worker of workers) {
      workerInitSignals.push(send({ type: MessageType.INIT, data: [worker.id, forkCount + 1] }, worker));
    }
    await Promise.all(workers);
    // await sendall({ type: MessageType.INIT, data: })
    _procCount = forkCount + 1;
    _id = 0;
  } else {
    // wait for message
    const data = (await receive()).data! as number[];
    // await send({ type: MessageType.INIT, data: []}, process);
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
  const results: Promise<void>[] = [];
  workers.forEach((worker) => {
    results.push(send(msg, worker));
  });
  return Promise.all(results);
};

export const receiveFromWorker = async (worker: cluster.Worker): Promise<Message> => {
  return new Promise((resolve) => {
    const listener = (m: Message) => {
      // console.log("Listener was called", m.data);
      resolve(m);
      messageBuffer.shift()!;
    };
    worker.once('message', listener);

    if (messageBuffer.length > 0) {
      const msg = messageBuffer.shift()!;
      // console.log("Listener skipped", msg.data)
      resolve(msg);
      worker.removeListener('message', listener);
    }
  });
};

export const receiveAll = async () => {
  const results: Promise<Message>[] = [];
  workers.forEach((worker) => {
    results.push(receiveFromWorker(worker));
  });
  return Promise.all(results);
};

export const receive = async (): Promise<Message> => {
  // TODO: can this fail?
  return new Promise((resolve) => {
    const listener = (m: Message) => {
      // console.log("Listener was called", m.data);
      resolve(m);
      messageBuffer.shift()!;
    };
    process.once('message', listener);

    if (messageBuffer.length > 0) {
      const msg = messageBuffer.shift()!;
      // console.log("Listener skipped", msg.data)
      resolve(msg);
      process.removeListener('message', listener);
    }
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
