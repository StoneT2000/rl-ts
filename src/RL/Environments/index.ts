import { NotImplementedError } from '../Errors';
import { Space } from '../Spaces';
export type RenderModes = 'human' | 'ansi' | 'rgb_array';

export type Dynamics<State, Action> = (sucessorState: State, reward: number, state: State, action: Action) => number;

// Extraction types to extract the generic type used in any environment

export type ExtractActionSpaceType<Env> = Env extends Environment<infer T, any, any, any, any> ? T : never;
export type ExtractObservationSpaceType<Env> = Env extends Environment<any, infer T, any, any, any> ? T : never;
export type ExtractActionType<Env> = Env extends Environment<any, any, infer T, any, any> ? T : never;
export type ExtractStateType<Env> = Env extends Environment<any, any, any, infer T, any> ? T : never;
export type ExtractRewardType<Env> = Env extends Environment<any, any, any, any, infer T> ? T : never;

export abstract class Environment<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State,
  Reward
> {
  constructor() {
    process.on("exit", () => {
      this.close();
    });
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
   * Resets the environment
   */
  abstract reset(): State;

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
   * @param sucessorState - s' - the succeeding state
   * @param reward - r - the reward returned upon transitioning to s' from s using action a
   * @param state - s - the preceeding state
   * @param action - a - action a to be taken
   */
  public dynamics(sucessorState: State, reward: number, state: State, action: Action): number {
    throw new NotImplementedError("Environment dynamics not implemented / provided");
  }

  /**
   * Environements can override this function to let users seed environments with a number
   * @param seed - seed number
   */
  public seed(seed: number): void {
    return;
  }

  /** Environments can override this function to let users clean up an environment. This is automatically run whenever a process exits */
  public close(): void {
    return;
  }

  public abstract actionSpace: ActionSpace;
  public abstract observationSpace: ObservationSpace;
}

export * as Examples from './examples';
