import { Environment, RenderModes } from '../..';
import path from 'path';
import { Box, Discrete } from '../../../Spaces';
import nj, { NdArray } from 'numjs';
import * as random from '../../../utils/random';
import { sleep } from '../../../utils/sleep';

export type State = NdArray<number>;
export type Action = number;
export type ActionSpace = Discrete;
export type ObservationSpace = Box;
export type Reward = number;

export interface CartPoleConfigs {
  maxEpisodeSteps: number;
}

/**
 * Simple GridWorld based on the gridworld presented in Chapters 3-4 in the Intro to RL book by Barto Sutton
 */
export class CartPole extends Environment<ObservationSpace, ActionSpace, State, Action, Reward> {
  public observationSpace: ObservationSpace;
  /** 0, 1, 2, 3 represent North, East, South, West directions */
  public actionSpace = new Discrete(4);

  public gravity = 9.8;
  public masscart = 1.0;
  public masspole = 0.1;
  public total_mass = this.masspole + this.masscart;
  public length = 0.5;
  public polemass_length = this.masspole * this.length;
  public force_mag = 10.0;
  public tau = 0.02;
  public theta_threshold_radians = (12 * 2 * Math.PI) / 360;
  public x_threshold = 2.4;
  public state: NdArray = random.random([4]);
  public steps_beyond_done: null | number = null;
  public timestep = 0;
  public maxEpisodeSteps = 500;
  public globalTimestep = 0;

  constructor(configs: Partial<CartPoleConfigs> = {}) {
    super('CartPole');
    if (configs.maxEpisodeSteps) {
      this.maxEpisodeSteps = configs.maxEpisodeSteps;
    }

    const caps = nj.array(
      [this.x_threshold * 2, Number.MAX_SAFE_INTEGER, this.theta_threshold_radians * 2, Number.MAX_SAFE_INTEGER],
      'float32'
    );
    this.observationSpace = new Box(caps.multiply(-1), caps, caps.shape, 'float32');
    this.actionSpace = new Discrete(2);
  }
  reset(): State {
    this.state = random.random([4], -0.05, 0.05);
    this.steps_beyond_done = null;
    this.timestep = 0;
    return this.state;
  }
  step(action: Action) {
    let reward = 1;
    const info = {};
    let x = this.state.get(0);
    let x_dot = this.state.get(1);
    let theta = this.state.get(2);
    let theta_dot = this.state.get(3);

    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);

    const temp = (force + this.polemass_length * (theta_dot * theta_dot) * sintheta) / this.total_mass;
    const thetaacc =
      (this.gravity * sintheta - costheta * temp) /
      (this.length * (4.0 / 3.0 - (this.masspole * (costheta * costheta)) / this.total_mass));
    const xacc = temp - (this.polemass_length * thetaacc * costheta) / this.total_mass;

    x = x + this.tau * x_dot;
    x_dot = x_dot + this.tau * xacc;
    theta = theta + this.tau * theta_dot;
    theta_dot = theta_dot + this.tau * thetaacc;

    this.state = nj.array([x, x_dot, theta, theta_dot]);
    this.timestep += 1;
    this.globalTimestep += 1;

    const done =
      x < -this.x_threshold ||
      x > this.x_threshold ||
      theta < -this.theta_threshold_radians ||
      theta > this.theta_threshold_radians ||
      this.timestep >= this.maxEpisodeSteps;

    if (!done) {
      reward = 1.0;
    } else if (this.steps_beyond_done === null) {
      // pole fell
      this.steps_beyond_done = 0;
      reward = 1.0;
    } else {
      if (this.steps_beyond_done === 0) {
        console.error(
          "You are calling 'step()' even though this environment already returned done = true. You should always call reset() once you receive 'done = true' -- any further steps are undefined behavior"
        );
      }
      this.steps_beyond_done += 1;
      reward = 0.0;
    }

    if (!this.actionSpace.contains(action)) {
      throw new Error(`${action} is invalid action in cartpole env`);
    }
    return {
      observation: this.state,
      reward,
      done,
      info,
    };
  }

  async render(
    mode: RenderModes,
    configs: { fps: number; episode?: number; rewards?: number } = { fps: 60 }
  ): Promise<void> {
    if (mode === 'human') {
      if (!this.viewer.isInitialized()) await this.viewer.initialize(path.join(__dirname, '../'), 'cartpole/');
      const delayMs = 1 / (configs.fps / 1000);
      await sleep(delayMs);
      await this.updateViewer(this.state, {
        timestep: this.timestep,
        globalTimestep: this.globalTimestep,
        x_threshold: this.x_threshold,
        pole_length: this.length,
        ...configs,
      });
    } else {
      throw new Error('Method not implemented.');
    }
  }
}
