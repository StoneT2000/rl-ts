import { Agent } from '../Agent';
import { Environment } from '../Environments';
import { Space } from '../Spaces';

//TODO: Handle stochastic environments
export class IterativePolicyEvaluation<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State
> {
  public env: Environment<ActionSpace, ObservationSpace, Action, State, number>;
  public valueFunction: Map<any, number> = new Map();
  public valueActionFunction: Map<any, { value: number; action: Action }> = new Map();
  public dynamics: (sucessorState: State, reward: number, state: State, action: Action) => number;
  constructor(
    env: Environment<ActionSpace, ObservationSpace, Action, State, number>,
    /** Function to map environment to a hashable state representation */
    public envToStateRep: (envToConvert: any) => any,
    /** Function to map state representation to a usable environment of the same class as this evaluator was constructed with */
    public envFromStateRep: (stateString: any) => typeof env,
    /** A list of all possible state representations */
    public allStateReps: any[],
    /** The policy function to evaluate */
    public policy: (action: Action, observation: State) => number,
    /** A list of all possible valid actions */
    public allPossibleActions: Action[],
    /** The dynamics of the environment. Does not to be given if environment has predefined dynamics */
    dynamics?: (sucessorState: State, reward: number, state: State, action: Action) => number,
  ) {
    this.env = env;
    allStateReps.forEach((s) => {
      this.valueFunction.set(s, 0);
    });
    if (!dynamics) {
      this.dynamics = this.env.dynamics;
    } else {
      this.dynamics = dynamics;
    }
  }
  /**
   * Estimates the value function of the given policy
   * @param params - the parameters object
   * @param params.epsilon - how accurate estimates are calculated - @default `1e-3`
   * @param params.steps - if set to positive integer, will train for this many number of steps regardless of epsilon choice - @default `undefined`
   * @param params.verbose - whether to log steps - @default `false`
   */
  train(
    params: {
      epsilon?: number;
      verbose: boolean;
      steps?: number;
    } = { epsilon: 1e-3, verbose: false }
  ): void {
    const { epsilon, verbose, steps } = params;
    for (let step = 1; ; step++) {
      if (steps && step > steps) {
        // stop training if steps is provided and step > steps
        break;
      }
      let updated_values = new Map();
      if (verbose) {
        console.log(`Step ${step}`);
      }
      let delta = 0;
      for (let stateString of this.allStateReps) {
        let val = 0;
        let s = this.envFromStateRep(stateString);
        let v_pi_s = 0;
        for (let action of this.allPossibleActions) {
          let observation = s.reset();
          let stepOut = s.step(action);
          let p_srsa = this.policy(action, observation);
          let reward = stepOut.reward;
          let done = stepOut.done;

          let sp_stateString = this.envToStateRep(s);

          let v_pi_sp = this.valueFunction.get(sp_stateString)!;

          // bind dynamics function to the current used environment
          this.dynamics = this.dynamics.bind(s);
          let p_sp_s_r = this.dynamics(stepOut.observation, reward, observation, action);
          v_pi_s += p_srsa * p_sp_s_r * (reward + 1 * v_pi_sp);
        }

        // calculate max delta to determine stopping condition
        let v_pi_s_old_val = this.valueFunction.get(stateString)!;
        delta = Math.max(delta, Math.abs(v_pi_s_old_val - v_pi_s));

        updated_values.set(stateString, v_pi_s);
      }
      updated_values.forEach((v, k) => {
        this.valueFunction.set(k, v);
      });

      if (steps === undefined && delta < epsilon!) {
        // stop training if steps are not defined and delta is small enough
        break;
      }
    }
  }
}
