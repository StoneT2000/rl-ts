import { Agent } from '../Agent';
import { Dynamics, Environment, RepToState, StateToRep } from '../Environments';
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

  public valueFunction: Map<any, number> = new Map();

  public policyStateToActionMap: Map<any, Action> = new Map();
  private rng: prng = seedrandom('');

  private dynamics: Dynamics<State, Action, number>;
  private stateToRep: StateToRep<State, any>;
  private repToState: RepToState<State, any>;

  /** Private sampleEnv for purposes of binding */
  private sampleEnv: Environment<ObservationSpace, ActionSpace, State, Action, number>;

  constructor(
    /** Function that creates a new environment that can be reset to different states */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, State, Action, number>,
    public configs: {
      /** Function to map environment to a hashable state representation. Required if environment does not provide this */
      stateToRep?: StateToRep<State, any>;
      /** Function to map state representation to a state to reset an environment to. Required if environment does not provide this */
      repToState?: RepToState<State, any>;
      /** A list of all possible state representations */
      allStateReps: any[];
      /** A list of all possible valid actions */
      allPossibleActions: Action[];
      /** The dynamics of the environment. Does not to be given if environment has predefined dynamics */
      dynamics?: Dynamics<State, Action, number>;
      /** Learning discount rate. Should be set in the range (0, 1) */
      discountRate: number;
      evaluatorConfigs: {
        epsilon: number;
        steps?: number;
      };
    }
  ) {
    super();
    // create an env to test it can be created and store relevant functions
    const env = makeEnv();
    this.sampleEnv = env;
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
    const rep = this.stateToRep.bind(this.sampleEnv)(observation);
    const a = this.policyStateToActionMap.get(rep)!;
    return a;
  }

  private evaluatePolicy() {
    const updated_values = new Map();
    for (let step = 1; ; step++) {
      let delta = 0;
      if (this.configs.evaluatorConfigs.steps && step >= this.configs.evaluatorConfigs.steps) {
        break;
      }
      for (const rep of this.configs.allStateReps) {
        const env = this.makeEnv();
        const state = this.repToState.bind(env)(rep);
        const obs = env.reset(state);
        const old_v = this.valueFunction.get(rep)!;

        const action = this.policy(obs);
        const stepOut = env.step(action);
        const reward = stepOut.reward;
        // TODO for stochastic environments, we need to iterate over all possible future states
        let p_sp_r_s_a = 0;

        p_sp_r_s_a = this.dynamics.bind(env)(stepOut.observation, reward, obs, action);

        const sp_stateString = this.stateToRep(stepOut.observation);
        const v = this.valueFunction.get(sp_stateString)!;
        const new_v = p_sp_r_s_a * (reward + this.configs.discountRate * v);
        updated_values.set(rep, new_v);
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
      for (const rep of this.configs.allStateReps) {
        const env = this.makeEnv();
        const state = this.repToState.bind(env)(rep);
        const obs = env.reset(state);
        const oldAction = this.policy(obs);
        let greedyAction = oldAction;
        let bestValue: number = Number.MIN_SAFE_INTEGER;

        // argmax action over all s', r : p(s', r | s, a) * [r + gamma * V(s')]
        for (const action of this.configs.allPossibleActions) {
          const state = this.repToState.bind(env)(rep);
          const obs = env.reset(state);
          const stepOut = env.step(action);
          const reward = stepOut.reward;
          let p_sp_r_s_a = this.dynamics.bind(env)(stepOut.observation, reward, obs, action);

          const sp_stateString = this.stateToRep(stepOut.observation);
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
        this.policyStateToActionMap.set(rep, greedyAction);
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
