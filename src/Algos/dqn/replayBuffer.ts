import Denque from 'denque';
import * as random from 'rl-ts/lib/utils/random';
/**
 * Defines a transition object, storing state, action, next state, and reward
 */
export interface Transition<Observation, Action> {
  state: Observation;
  action: Action;
  reward: number;
  nextState: Observation;
  done: boolean;
}

export class ReplayBuffer<Observation, Action> {
  public memory: Denque<Transition<Observation, Action>>;
  constructor(
    /** Capacity of the replay memory */
    public capacity: number
  ) {
    this.memory = new Denque([], { capacity });
  }
  public push(transition: Transition<Observation, Action>): void {
    this.memory.push(transition);
  }
  public sample(batchSize: number): Array<Transition<Observation, Action>> {
    // TODO: use resovoir sampling for picking unique k
    const sample: Array<Transition<Observation, Action>> = [];
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
