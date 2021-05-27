import { Environment, RenderModes } from '..';
import { Box, Discrete } from '../../Spaces';
import nj, { NdArray } from 'numjs';
import * as random from '../../utils/random';
import { Scalar } from '@tensorflow/tfjs-core';

export type State = NdArray<number>;
export type Action = number;
export type ActionSpace = Discrete;
export type ObservationSpace = Box;
export type Reward = number;

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
  public kinematics_integrator = 'euler';
  public theta_threshold_radians = (12 * 2 * Math.PI) / 360;
  public x_threshold = 2.4;
  public state: NdArray = random.random([4]);
  public steps_beyond_done: null | number = null;
  constructor() {
    super();

    let caps = nj.array(
      [this.x_threshold * 2, Number.MAX_SAFE_INTEGER, this.theta_threshold_radians * 2, Number.MAX_SAFE_INTEGER],
      'float32'
    );
    this.observationSpace = new Box(caps.multiply(-1), caps, caps.shape, 'float32');
    this.actionSpace = new Discrete(2);
  }
  reset(): State {
    this.state = random.random([4], -0.05, 0.05);
    this.steps_beyond_done = null;
    return this.state;
  }
  step(action: Action) {
    let reward = 1;
    let info = {};
    let x = this.state.get(0);
    let x_dot = this.state.get(1);
    let theta = this.state.get(2);
    let theta_dot = this.state.get(3);

    let force = action === 1 ? this.force_mag : -this.force_mag;
    let costheta = Math.cos(theta);
    let sintheta = Math.sin(theta);

    let temp = (force + this.polemass_length * theta_dot ** 2 * sintheta) / this.total_mass;
    let thetaacc =
      (this.gravity * sintheta - costheta * temp) /
      (this.length * (4.0 / 3.0 - (this.masspole * costheta ** 2) / this.total_mass));
    let xacc = temp - (this.polemass_length * thetaacc * costheta) / this.total_mass;

    if (this.kinematics_integrator === 'euler') {
      x = x + this.tau * x_dot;
      x_dot = x_dot + this.tau * xacc;
      theta = theta + this.tau * theta_dot;
      theta_dot = theta_dot + this.tau * thetaacc;
    } else {
      x_dot = x_dot + this.tau * xacc;
      x = x + this.tau * x_dot;
      theta_dot = theta_dot + this.tau * thetaacc;
      theta = theta + this.tau * theta_dot;
    }
    this.state = nj.array([x, x_dot, theta, theta_dot]);

    const done = x < -this.x_threshold || x > this.x_threshold || theta < -this.theta_threshold_radians || theta > this.theta_threshold_radians;

    if (!done) {
      reward = 1.0;
    } else if (this.steps_beyond_done === null) {
      // pole fell
      this.steps_beyond_done = 0;
      reward = 1.0;
    } else {
      if (this.steps_beyond_done === 0) {
        console.error("You are calling 'step()' even though this environment already returned done = true. You should always call reset() once you receive 'done = true' -- any further steps are undefined behavior")
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

  render(mode: RenderModes): void {
    throw new Error('Method not implemented.');
  }
}
