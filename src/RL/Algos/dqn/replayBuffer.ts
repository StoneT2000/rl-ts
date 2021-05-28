import Denque from 'denque';
import * as random from '../../utils/random';
/**
 * Defines a transition object, storing state, action, next state, and reward
 */
export interface Transition<State, Action> {
  state: State;
  action: Action;
  reward: number;
  nextState: State;
  done: boolean;
}

export class ReplayBuffer<State, Action> {
  public memory: Denque<Transition<State, Action>>;
  constructor(
    /** Capacity of the replay memory */
    public capacity: number
  ) {
    this.memory = new Denque([], { capacity });
  }
  public push(transition: Transition<State, Action>): void {
    this.memory.push(transition);
  }
  public sample(batchSize: number): Array<Transition<State, Action>> {
    // TODO: use resovoir sampling for picking unique k
    const sample: Array<Transition<State, Action>> = [];
    for (let i = 0; i < batchSize; i++) {
      // this.memory.length
      const k = Math.floor(this.memory.length * random.randomVal());
      sample.push(this.memory.get(k)!);
    }
    return sample;
  }
  public get length() {
    return this.memory.length;
  }
}
