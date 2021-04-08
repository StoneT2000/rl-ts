
/**
 * Defines a generic mdp agent
 */
export abstract class Agent<State, Action> {
  /**
   * Selects an action given the new observation
   * @param observation 
   */
  abstract action(observation: State): Action;
}