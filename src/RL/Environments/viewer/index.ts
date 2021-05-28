import http from 'http';
import { Server } from 'socket.io';
import express from 'express';
import open from 'open';
export class Viewer<State> {
  io: Server | undefined;
  _socketConnections = 0;
  private connectPromise: Promise<void> | undefined;
  private resolveConnectPromise: Function = () => {};
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
      this.setupConnectPromise();

      const app = express();
      const server = new http.Server(app);
      this.io = new Server(server);
      const port = process.env.PORT || 3000;
      
      

      app.use(express.static(distPath));

      this.io.on('connection', (socket) => {
        this._socketConnections += 1;
        socket.on('disconnect', () => {
          this._socketConnections -= 1;
          if (this._socketConnections === 0) {
            this.setupConnectPromise();
          }
        });
        this.resolveConnectPromise();
        this.connectPromise = undefined;
        resolve();
      });
      server
        .listen(port, () => {
          console.log(`Viewer running at http://localhost:${port}/${urlPath}`);
          
          open(`http://localhost:${port}/${urlPath}`);
        })
        .on('error', (err) => {
          reject(err);
        });
    });
  }
  /** step forward in the viewer by emitting data provided a client is connected */
  async step(state: State, info: any) {
    if (!this.io) {
      throw new Error("Socket not initialized");
    }
    if (this.connectPromise) {
      console.log("Viewer paused, waiting for web viewer to be opened again");
      await this.connectPromise;
      console.log("Viewer resumed");
    }
    this.io.emit('data', state, info);
  }

  private setupConnectPromise() {
    this.connectPromise = new Promise((res) => {this.resolveConnectPromise = res});
  }
}
