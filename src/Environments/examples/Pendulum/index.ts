import { Environment, RenderModes } from 'rl-ts/lib/Environments';
import path from 'path';
import { Box, Discrete } from 'rl-ts/lib/Spaces';
import nj, { NdArray } from 'numjs';
import * as random from 'rl-ts/lib/utils/random';
import { tensorLikeToNdArray } from 'rl-ts/lib/utils/np';

/** Vector with shape (2, ) */
export type State = NdArray<number>;
/** Vector with shape (3, ) */
export type Action = NdArray<number> | number;
export type Observation = NdArray<number>;
export type ObservationSpace = Box;
export type Reward = number;

export interface PendulumConfigs {
  /** gravity constant */
  g: number;
  /** simulation accuracy, lower the more fine grained calculations are */
  dt: number;
  /** Whether or not to discretize the action space */
  discretizeActionSpace: boolean;
}

/**
 * Pendulum Environment
 */
export class Pendulum extends Environment<ObservationSpace, any, Observation, State, Action, Reward> {
  public observationSpace: ObservationSpace;
  public max_speed = 8;
  public max_torque = 2;
  public dt = 0.05;
  public g = 9.81;
  public m = 1;
  public l = 1;
  public actionSpace: Box | Discrete;
  public state: NdArray = random.random([2]);

  public maxEpisodeSteps = 500;
  public timestep = 0;

  private last_u: number | undefined;

  constructor(configs: Partial<PendulumConfigs> = {}) {
    super('Pendulum');
    if (configs.g) {
      this.g = configs.g;
    }
    if (configs.dt) {
      this.dt = configs.dt;
    }

    const caps = nj.array([1, 1, this.max_speed], 'float32');
    this.observationSpace = new Box(caps.multiply(-1), caps, caps.shape, 'float32');
    if (configs.discretizeActionSpace) {
      this.actionSpace = new Discrete(2);
    } else {
      this.actionSpace = new Box(-this.max_torque, this.max_torque, [1], 'float32');
    }
  }
  reset(): Promise<Observation> {
    const high = nj.array([Math.PI, 1]);
    this.state = random.random([2], 0, 1).multiply(high.multiply(2)).subtract(high);
    this.timestep = 0;
    return Promise.resolve(this.getObs());
  }
  async step(action: Action) {
    const th = this.state.get(0);
    const thdot = this.state.get(1);

    const g = this.g;
    const m = this.m;
    const l = this.l;
    const dt = this.dt;
    let u = tensorLikeToNdArray(action);
    if (this.actionSpace.meta.discrete) {
      if (u.get(0) === 1) {
        u = nj.array([this.max_torque]);
      } else {
        u = nj.array([-this.max_torque]);
      }
    } else {
      u = nj.clip(u, -this.max_torque, this.max_torque).slice([0, 1]);
    }

    this.last_u = u.get(0, 0); // for rendering
    const costs = this.angleNormalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u.get(0, 0) ** 2;

    let newthdot = thdot + (((-3 * g) / (2 * l)) * Math.sin(th + Math.PI) + (3 / (m * l ** 2)) * u.get(0, 0)) * dt;
    const newth = th + newthdot * dt;
    newthdot = nj.clip(newthdot, -this.max_speed, this.max_speed).get(0, 0);

    this.state = nj.array([newth, newthdot]);
    this.timestep += 1;
    return {
      observation: this.getObs(),
      reward: -costs,
      done: this.timestep >= this.maxEpisodeSteps,
      info: {},
    };
  }
  private getObs() {
    const th = this.state.get(0);
    const thdot = this.state.get(1);
    return nj.array([Math.cos(th), Math.sin(th), thdot]);
  }
  private angleNormalize(th: number) {
    return ((th + Math.PI) % (2 * Math.PI)) - Math.PI;
  }

  async render(
    mode: RenderModes,
    configs: { fps: number; episode?: number; rewards?: number } = { fps: 60 }
  ): Promise<void> {
    if (mode === 'web') {
      if (!this.viewer.isInitialized()) await this.viewer.initialize(path.join(__dirname, '../'), 'pendulum/');
      const delayMs = 1 / (configs.fps / 1000);
      await this.sleep(delayMs);
      await this.updateViewer(this.state, {
        last_u: this.last_u,
        ...configs,
      });
    } else {
      throw new Error(`${mode} is not an available render mode`);
    }
  }
}
