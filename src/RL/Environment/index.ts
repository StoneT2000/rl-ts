import { Space } from "../Spaces";

export abstract class Environment<ActionSpace extends Space<Action>, ObservationSpace extends Space<State>, Action, State, Reward> {

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
  abstract step(action: Action): {observation: State, reward: Reward, done: boolean};

  /**
   * Resets the environment
   */
  abstract reset(): void;

  public abstract actionSpace: ActionSpace;
  public abstract observationSpace: ObservationSpace;
}