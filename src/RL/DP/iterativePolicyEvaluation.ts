import { Agent } from '../Agent';
import { Environment } from '../Environment';
import { Space } from '../Spaces';

export class IterativePolicyEvaluation<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State
> extends Agent<State, Action> {
  public env: Environment<ActionSpace, ObservationSpace, Action, State, number>;
  public valueFunction: Map<string, number> = new Map();
  public valueActionFunction: Map<string, { value: number; action: Action }> = new Map();
  constructor(env: Environment<ActionSpace, ObservationSpace, Action, State, number>) {
    super();
    this.env = env;
  }
  train(steps: number): void {}
  action(observation: State): Action {
    let hash = this.hashState(observation);
    let choice = this.valueActionFunction.get(hash);
    if (!choice) return this.env.actionSpace.sample();
    return choice.action;
  }
  private hashState(observation: State): string {
    return JSON.stringify(this.env.observationSpace.to_jsonable([observation])[0]);
  }
}
