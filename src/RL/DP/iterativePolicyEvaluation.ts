import { Dynamics, Environment, RepToState, StateToRep } from '../Environments';
import { Space } from '../Spaces';

//TODO: Handle stochastic environments

export class IterativePolicyEvaluation<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State
> {
  public valueFunction: Map<any, number> = new Map();
  // TODO: public valueActionFunction: Map<any, { value: number; action: Action }> = new Map();
  private dynamics: Dynamics<State, Action, number>;
  private stateToRep: StateToRep<State, any>;
  private repToState: RepToState<State, any>;
  constructor(
    /** Function that creates a new environment that can be reset to different states */
    public makeEnv: () => Environment<ActionSpace, ObservationSpace, Action, State, number>,
    public configs: {
      /** Function to map environment to a hashable state representation. Required if environment does not provide this */
      stateToRep?: StateToRep<State, any>;
      /** Function to map state representation to a state to reset an environment to. Required if environment does not provide this */
      repToState?: RepToState<State, any>;
      /** A list of all possible state representations */
      allStateReps: any[];
      /** The policy function to evaluate */
      policy: (action: Action, observation: State) => number;
      /** A list of all possible valid actions */
      allPossibleActions: Action[];
      /** The dynamics of the environment. Required if environment does not provide this */
      dynamics?: Dynamics<State, Action, number>;
    }
  ) {
    this.configs.allStateReps.forEach((s) => {
      this.valueFunction.set(s, 0);
    });
    // create an env to test it can be created and store relevant functions
    const env = makeEnv();
    if (!this.configs.dynamics) {
      this.dynamics = env.dynamics;
    } else {
      this.dynamics = this.configs.dynamics;
    }
    if (!this.configs.stateToRep) {
      this.stateToRep = env.stateToRep;
    } else {
      this.stateToRep = this.configs.stateToRep;
    }
    if (!this.configs.repToState) {
      this.repToState = env.repToState;
    } else {
      this.repToState = this.configs.repToState;
    }
  }
  setPolicy(policy: (action: Action, observation: State) => number): void {
    this.configs.policy = policy;
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
      const updated_values = new Map();
      if (verbose) {
        console.log(`Step ${step}`);
      }
      let delta = 0;
      for (const rep of this.configs.allStateReps) {
        const env = this.makeEnv();
        let v_pi_s = 0;
        for (const action of this.configs.allPossibleActions) {
          const state = this.repToState.bind(env)(rep);
          const observation = env.reset(state);
          const stepOut = env.step(action);
          const p_srsa = this.configs.policy(action, observation);
          const reward = stepOut.reward;

          // note, we bind stateToRep because this function could be pulled from the environment and use the this keyword in it
          const sp_stateRep = this.stateToRep.bind(env)(stepOut.observation);

          const v_pi_sp = this.valueFunction.get(sp_stateRep)!;

          // bind dynamics function to the current used environment

          let p_sp_s_r = 0;
          p_sp_s_r = this.dynamics.bind(env)(stepOut.observation, reward, observation, action);

          v_pi_s += p_srsa * p_sp_s_r * (reward + 1 * v_pi_sp);
        }

        // calculate max delta to determine stopping condition
        const v_pi_s_old_val = this.valueFunction.get(rep)!;
        delta = Math.max(delta, Math.abs(v_pi_s_old_val - v_pi_s));

        updated_values.set(rep, v_pi_s);
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
