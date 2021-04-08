import { Space } from '../Spaces';
export type RenderModes = 'human' | 'ansi' | 'rgb_array';
export abstract class Environment<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State,
  Reward
> {
  /**
   * Step forward in time by one time step and process the action
   * @param action - the action to perform in the environment
   * @returns the new observed state, reward, and whether a terminal state has been reached (done). Specifically return type is
   * {
   *  observation: State,
   *  reward: Reward,
   *  done: boolean
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

  public abstract actionSpace: ActionSpace;
  public abstract observationSpace: ObservationSpace;
}

export * as Examples from './examples';
