import { expect } from 'chai';
import { DP } from '../../src';
import { IterativePolicyEvaluation } from '../../src/RL/DP/iterativePolicyEvaluation';
import { SimpleGridWorld } from '../../src/RL/Environments/examples';

describe('Test Iterative Policy Evaluation', () => {
  it('Should solve simple grid world for equiprobable policy', () => {
    let width = 4;
    let height = 4;
    let targetPositions = [
      { x: 3, y: 3 },
      { x: 0, y: 0 },
    ];
    let env = new SimpleGridWorld(width, height, targetPositions, { x: 1, y: 0 });

    // the equiprobably policy
    let policy = (action: number, state: typeof env.state) => {
      return 0.25;
    };

    // this definition of dynamics is sufficient as the environment is completely deterministic and we know many priors
    // To be more rigorous, you could check the actual probability of reaching successorState by applying the given action to state
    let dynamics = (sucessorState: typeof env.state, reward: number, state: typeof env.state, action: number) => {
      if (reward == 0) return 1;
      else {
        if (env.posIsInTargetPositions(state.agentPos)) {
          return 0;
        }
      }
      return 1;
    };

    let envToStateRep = (env: SimpleGridWorld) => {
      return env.state.agentPos.x + env.state.agentPos.y * Math.max(width, height);
    };
    let envFromStateRep = (stateString: string) => {
      let hash = parseInt(stateString);
      let m = Math.max(width, height);
      let x = hash % m;
      let y = Math.floor(hash / m);
      return new SimpleGridWorld(width, height, targetPositions, { x, y });
    };
    let allStateReps = [];
    for (let x = 0; x < env.width; x++) {
      for (let y = 0; y < env.height; y++) {
        let pos = { x: x, y: y };
        let env = new SimpleGridWorld(width, height, targetPositions, pos);
        allStateReps.push(envToStateRep(env));
      }
    }
    let policyEvaluator = new DP.IterativePolicyEvaluation(
      env,
      //@ts-ignore;
      envToStateRep,
      envFromStateRep,
      allStateReps,
      policy,
      dynamics,
      [0, 1, 2, 3]
    );
    policyEvaluator.train(10);

    let gtVals = [
      0.0,
      -6.14,
      -8.35,
      -8.97,
      -6.14,
      -7.74,
      -8.43,
      -8.35,
      -8.35,
      -8.43,
      -7.74,
      -6.14,
      -8.97,
      -8.35,
      -6.14,
      0.0,
    ];
    for (let i = 0; i <= 15; i++) {
      expect(policyEvaluator.valueFunction.get(i)).to.approximately(gtVals[i], 1e-2);
    }
  });
});
