import { Agent } from '../Agent';
import { Environment } from '../Environments';
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
  public dynamics: null | ((sucessorState: State, reward: number, state: State, action: Action) => number);
  public valueFunction: Map<any, number> = new Map();

  public policyStateToActionMap: Map<any, Action> = new Map();
  private rng: prng = seedrandom('');

  constructor(
    public configs: {
      /** Function to map environment to a hashable state representation */
      envToStateRep: (envToConvert: any) => any;
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
      }
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
      let a = this.configs.allPossibleActions[this.rng() * this.configs.allPossibleActions.length];
      this.policyStateToActionMap.set(s, a);
    });
  }
  seed(seed: number) {
    this.rng = seedrandom(`${seed}`);
  }
  policy(observation: State): Action {
    let s = this.configs.envFromStateRep(observation);
    let stateRep = this.configs.envToStateRep(s);
    let a = this.policyStateToActionMap.get(stateRep)!;
    return a;
  };

  private evaluatePolicy() {
    let delta = 0
    let updated_values = new Map();
    for (let step = 1;; step++) {
      if (this.configs.evaluatorConfigs.steps && step >= this.configs.evaluatorConfigs.steps) {
        break;
      }
      for (const stateRep of this.configs.allStateReps) {
        let env = this.configs.envFromStateRep(stateRep);
        let obs = env.reset()
        let old_v = this.valueFunction.get(stateRep)!;
        let action = this.policy(obs);
        let stepOut = env.step(action);
        let reward = stepOut.reward;
        // TODO for stochastic environments, we need to iterate over all possible future states
        let p_sp_r_s_a = 0;
        if (this.dynamics) {
          p_sp_r_s_a = this.dynamics(stepOut.observation, reward, obs, action);
        } else {
          p_sp_r_s_a = env.dynamics(stepOut.observation, reward, obs, action);
        }
        let sp_stateString = this.configs.envToStateRep(env);
        let v = this.valueFunction.get(sp_stateString)!;
        let new_v = p_sp_r_s_a * (reward + this.configs.discountRate * v);
        
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

  train(params: {
    untilStable: boolean;
    verbose: boolean;
    steps?: number;
  } = {
    untilStable: true,
    verbose: true
  }) {
    let step = 1;
    while(true) {
      this.evaluatePolicy();
      let policyStable = true;
      for (const stateRep of this.configs.allStateReps) {
        let env = this.configs.envFromStateRep(stateRep);
        let obs = env.reset();
        let oldAction = this.policy(obs);
        let greedyAction = oldAction;
        let bestValue: number = Number.MIN_VALUE;

        // argmax action over all s', r : p(s', r | s, a) * [r + gamma * V(s')]
        for (const action of this.configs.allPossibleActions) {
          let p_sp_r_s_a = 0;
          let obs = env.reset();
          let stepOut = env.step(action);
          let reward = stepOut.reward;
          if (this.dynamics) {
            p_sp_r_s_a = this.dynamics(stepOut.observation, reward, obs, action);
          } else {
            p_sp_r_s_a = env.dynamics(stepOut.observation, reward, obs, action);
          }
          let sp_stateString = this.configs.envToStateRep(env);
          let v_sp = this.valueFunction.get(sp_stateString)!;
          let value = p_sp_r_s_a * (reward + this.configs.discountRate * v_sp);
          if (value > bestValue) {
            bestValue = value;
            greedyAction = action;
          }
        }

        if (greedyAction !== oldAction) {
          policyStable = false;
        }
      }
      
      step += 1;
      console.log(`Step ${step}`);
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
