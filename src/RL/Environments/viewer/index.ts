import http from 'http';
import { Server } from 'socket.io';
import express from 'express';
import open from 'open';
export class Viewer<State> {
  io: Server | undefined;
  constructor() {}
  isInitialized(): boolean {
    return this.io !== undefined;
  }
  /**
   * Initialize the viewer
   * @param scripts - scripts to load
   * @param stylesheets - stylesheets to load
   * @returns
   */
  async initialize(distPath: string, urlPath?: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const app = express();
      const server = new http.Server(app);
      this.io = new Server(server);
      const port = process.env.PORT || 3000;

      app.use(express.static(distPath));

      this.io.on('connection', () => {
        // console.log(`Socket connected`);
      });
      server
        .listen(port, () => {
          console.log(`Viewer running at http://localhost:${port}/`);
          open(`http://localhost:${port}/${urlPath}`);
          resolve();
        })
        .on('error', (err) => {
          reject(err);
        });
    });
  }
  step(data: State, info: any) {
    this.io!.emit('data', data, info);
  }
}
