/**
 * Defines a generic agent
 */
export abstract class Agent<Observation, Action> {
  /**
   * Selects an action given the new observation
   * @param observation
   */
  abstract act(observation: Observation): Action;

  /**
   * Override this function to let user's seed the agent's rng
   * @param seed
   */
  // eslint-disable-next-line
  public seed(seed: number): void {
    return;
  }
}
