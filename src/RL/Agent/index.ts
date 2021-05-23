/**
 * Defines a generic mdp agent
 */
export abstract class Agent<State, Action> {
  /**
   * Selects an action given the new observation
   * @param observation
   */
  abstract action(observation: State): Action;

  /**
   * Override this function to let user's seed the agent's rng
   * @param seed
   */
  // eslint-disable-next-line
  public seed(seed: number): void {
    return;
  }
}
