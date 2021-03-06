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
const messageBuffer: Record<number, Message[]> = {};
// export const setupMPI = async () => {
// TODO: do any setup here if needed
// };

export const fork = async (forkCount: number) => {
  const listeningPromises: Array<Promise<cluster.Worker>> = [];
  if (cluster.isMaster) {
    for (let i = 0; i < forkCount; i++) {
      const worker = cluster.fork();
      messageBuffer[worker.id] = [];
      listeningPromises.push(
        new Promise((res, rej) => {
          worker.on('online', async () => {
            res(worker);
          });
          worker.on('message', (m) => {
            messageBuffer[worker.id].push(m);
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
    _procCount = forkCount + 1;
    _id = 0;
    // sync with workers so all processes start together
    await receiveAll();
  } else {
    messageBuffer[0] = [];
    process.on('message', (m) => {
      messageBuffer[0].push(m);
    });
    // wait for message
    const data = (await receive()).data! as number[];
    _id = data[0];
    _procCount = data[1];

    // sync with primary so all processes start together
    await send({ type: MessageType.INIT, data: [] }, process);
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
      resolve(m);
      messageBuffer[worker.id].shift()!;
    };
    worker.once('message', listener);
    if (messageBuffer[worker.id].length > 0) {
      const msg = messageBuffer[worker.id].shift()!;
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

/** Used by workers to receive messages from primary */
export const receive = async (): Promise<Message> => {
  // TODO: can this fail?
  return new Promise((resolve) => {
    const listener = (m: Message) => {
      // console.log("Listener was called", m.data);
      resolve(m);
      messageBuffer[0].shift()!;
    };
    process.once('message', listener);

    if (messageBuffer[0].length > 0) {
      const msg = messageBuffer[0].shift()!;
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
