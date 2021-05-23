import { Agent } from '../Agent';
import { Dynamics, Environment } from '../Environments';
import { Space } from '../Spaces';
import seedrandom from 'seedrandom';
import { prng } from '../utils/random';

// TODO: enable stochastic environments
export class PolicyIteration<
  ActionSpace extends Space<Action>,
  ObservationSpace extends Space<State>,
  Action,
  State
> extends Agent<State, Action> {
  public dynamics: null | Dynamics<State, Action>;
  public valueFunction: Map<any, number> = new Map();

  public policyStateToActionMap: Map<any, Action> = new Map();
  private rng: prng = seedrandom('');

  constructor(
    public configs: {
      /** Function to map environment to a hashable state representation */
      obsToStateRep: (state: State) => any;
      /** Function to map state representation to a usable environment of the same class as this evaluator was constructed with */
      envFromStateRep: (stateString: any) => Environment<ActionSpace, ObservationSpace, Action, State, number>;
      /** A list of all possible state representations */
      allStateReps: any[];
      /** A list of all possible valid actions */
      allPossibleActions: Action[];
      /** The dynamics of the environment. Does not to be given if environment has predefined dynamics */
      dynamics?: (sucessorState: State, reward: number, state: State, action: Action) => number;
      /** Learning discount rate. Should be set in the range (0, 1) */
      discountRate: number;
      evaluatorConfigs: {
        epsilon: number;
        steps?: number;
      };
    }
  ) {
    super();
    if (!this.configs.dynamics) {
      this.dynamics = null;
    } else {
      this.dynamics = this.configs.dynamics;
    }

    // initialize value function and policy
    this.configs.allStateReps.forEach((s) => {
      this.valueFunction.set(s, 0);
      const k = Math.floor(this.rng() * this.configs.allPossibleActions.length);
      const a = this.configs.allPossibleActions[k];
      this.policyStateToActionMap.set(s, a);
    });
  }
  seed(seed: number): void {
    this.rng = seedrandom(`${seed}`);
  }
  policy(observation: State): Action {
    const stateRep = this.configs.obsToStateRep(observation);
    const a = this.policyStateToActionMap.get(stateRep)!;
    return a;
  }

  /**
   * Internal function used for retrieving the relevant dynamics function depending if user provided one or environment came with it.
   * @param env
   * @returns Dynamics Function
   */
  private getEnvDynamics(
    env: Environment<ActionSpace, ObservationSpace, Action, State, number>
  ): Dynamics<State, Action> {
    if (this.dynamics) return this.dynamics.bind(env);
    return env.dynamics.bind(env);
  }

  private evaluatePolicy() {
    const updated_values = new Map();
    for (let step = 1; ; step++) {
      let delta = 0;
      if (this.configs.evaluatorConfigs.steps && step >= this.configs.evaluatorConfigs.steps) {
        break;
      }
      for (const stateRep of this.configs.allStateReps) {
        const env = this.configs.envFromStateRep(stateRep);
        const obs = env.reset();
        const old_v = this.valueFunction.get(stateRep)!;

        const action = this.policy(obs);
        const stepOut = env.step(action);
        const reward = stepOut.reward;
        // TODO for stochastic environments, we need to iterate over all possible future states
        let p_sp_r_s_a = 0;

        p_sp_r_s_a = this.getEnvDynamics(env)(stepOut.observation, reward, obs, action);

        const sp_stateString = this.configs.obsToStateRep(stepOut.observation);
        const v = this.valueFunction.get(sp_stateString)!;
        const new_v = p_sp_r_s_a * (reward + this.configs.discountRate * v);
        updated_values.set(stateRep, new_v);
        delta = Math.max(delta, Math.abs(new_v - old_v));
      }
      updated_values.forEach((v, k) => {
        this.valueFunction.set(k, v);
      });
      if (delta < this.configs.evaluatorConfigs.epsilon) {
        break;
      }
    }
  }

  train(
    params: {
      untilStable: boolean;
      verbose: boolean;
      steps?: number;
    } = {
      untilStable: true,
      verbose: true,
    }
  ): void {
    for (let step = 1; ; step++) {
      this.evaluatePolicy();
      let policyStable = true;
      if (params.verbose) {
        console.log(`Step ${step}`);
      }
      for (const stateRep of this.configs.allStateReps) {
        const env = this.configs.envFromStateRep(stateRep);
        const obs = env.reset();
        const oldAction = this.policy(obs);
        let greedyAction = oldAction;
        let bestValue: number = Number.MIN_SAFE_INTEGER;

        // argmax action over all s', r : p(s', r | s, a) * [r + gamma * V(s')]
        for (const action of this.configs.allPossibleActions) {
          let p_sp_r_s_a = 0;
          const obs = env.reset();
          const stepOut = env.step(action);
          const reward = stepOut.reward;
          p_sp_r_s_a = this.getEnvDynamics(env)(stepOut.observation, reward, obs, action);

          const sp_stateString = this.configs.obsToStateRep(stepOut.observation);
          const v_sp = this.valueFunction.get(sp_stateString)!;
          const value = p_sp_r_s_a * (reward + this.configs.discountRate * v_sp);
          if (value > bestValue) {
            bestValue = value;
            greedyAction = action;
          }
        }
        if (greedyAction !== oldAction) {
          policyStable = false;
        }
        this.policyStateToActionMap.set(stateRep, greedyAction);
      }
      if (params.steps && step >= params.steps) {
        break;
      }
      if (policyStable) {
        break;
      }
    }
  }
  action(observation: State): Action {
    return this.policy(observation);
  }
}
