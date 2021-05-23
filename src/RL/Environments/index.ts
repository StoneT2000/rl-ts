import { NotImplementedError } from '../Errors';
import { Space } from '../Spaces';
export type RenderModes = 'human' | 'ansi' | 'rgb_array';

export type Dynamics<State, Action, Reward> = (
  sucessorState: State,
  reward: Reward,
  state: State,
  action: Action
) => number;

export type StateToRep<State, Rep> = (state: State) => Rep;
export type RepToState<State, Rep> = (rep: Rep) => State;

// Extraction types to extract the generic type used in any environment

export type ExtractActionSpaceType<Env> = Env extends Environment<infer T, any, any, any, any> ? T : never;
export type ExtractObservationSpaceType<Env> = Env extends Environment<any, infer T, any, any, any> ? T : never;
export type ExtractActionType<Env> = Env extends Environment<any, any, infer T, any, any> ? T : never;
export type ExtractStateType<Env> = Env extends Environment<any, any, any, infer T, any> ? T : never;
export type ExtractRewardType<Env> = Env extends Environment<any, any, any, any, infer T> ? T : never;

/**
 * @class Environment
 *
 * A class for defining an environment that is fully observable. This is enforced by requiring the observation space return data of type state.
 */
export abstract class Environment<
  ObservationSpace extends Space<State>,
  ActionSpace extends Space<Action>,
  State,
  Action,
  Reward
> {
  /**
   * Construct a new environment. NOTE: it is recommended to define any state related code in the reset()
   * function to keep the environment episodic. Even if the environment has infinite horizon,
   * this is still recommended */
  constructor() {
    // TODO: check if this is okay to do as a cleanup method
    // process.on("exit", () => {
    //   this.close();
    // });
  }

  /**
   * Step forward in time by one time step and process the action
   * @param action - the action to perform in the environment
   * @returns the new observed state, reward, and whether a terminal state has been reached (done) and any other info. Specifically return type is
   * {
   *  observation: State,
   *  reward: Reward,
   *  done: boolean,
   *  info: any
   * }
   */
  abstract step(action: Action): { observation: State; reward: Reward; done: boolean; info?: any };

  /**
   * Resets the environment to an initial state and return the initial observation
   *
   * Should always be called first prior to calling step
   *
   * @param state - a state to load the environment with instead of generating an initial state. Note, not all environments are guranteed to use this
   */
  abstract reset(state?: State): State;

  /**
   * Renders the environment
   * @param mode - The render mode to use. "human" is human readable output rendered to stdout. "ansi" returns a string containing terminal-style text representation. "rgb_array" returns an array representing rgb values per pixel in a image
   */
  abstract render(mode: RenderModes): void;

  /**
   * The dynamics of the environment. Throws an error when called if a environment does not implement this
   *
   * Mathematically defined as P(s', r | s, a) - the probability of transitioning to state s' from s and receiving reward r after taking action a.
   *
   * This should not change the environment in any way.
   *
   * Note: avoid using the "this" keyword in this function. While allowed, it may cause errors in some algorithms
   *
   * @param sucessorState - s' - the succeeding state
   * @param reward - r - the reward returned upon transitioning to s' from s using action a
   * @param state - s - the preceeding state
   * @param action - a - action a to be taken
   */
  // eslint-disable-next-line
  public dynamics(sucessorState: State, reward: number, state: State, action: Action): number {
    throw new NotImplementedError('Environment dynamics not implemented / provided');
  }

  /**
   * Hashes this environment's state into a hashable representation (rep).
   * This should not change the environment in any way.
   *
   * Note: avoid using the "this" keyword in this function. While allowed, it may cause errors in some algorithms
   *
   * @param state - the state to hash. Should be the same type as the state of the environment
   */
  // eslint-disable-next-line
  public stateToRep(state: State): any {
    throw new NotImplementedError('Environment hashable state function not implemented');
  }

  /**
   * Converts a hashable representation (rep) into a state object
   * This should not change the environment in any way.
   *
   * Note: avoid using the "this" keyword in this function. While allowed, it may cause errors in some algorithms
   * @param rep - the rep to convert to a state
   */
  // eslint-disable-next-line
  public repToState(rep: any): State {
    throw new NotImplementedError('Environment hashable state function not implemented');
  }

  /**
   * Environements can override this function to let users seed environments with a number
   * @param seed - seed number
   */
  // eslint-disable-next-line
  public seed(seed: number): void {
    return;
  }

  /** Environments can override this function to let users clean up an environment. This is automatically run whenever a process exits */
  public close(): void {
    return;
  }

  /** Defines the space of actions allowable */
  public abstract actionSpace: ActionSpace;
  /** Defines the space of observable observations. */
  public abstract observationSpace: ObservationSpace;

  /**
   * Fully define the current state of the environment.
   * Avoid storing anything that is not a JS primitive,
   * stick to just using arrays, strings, numbers, BigInt, booleans, and Symbols */
  public abstract state: State;
}

export * as Examples from './examples';
