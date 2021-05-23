import { expect } from 'chai';
import { DP } from '../../src';
import { ExtractStateType } from '../../src/RL/Environments';
import { SimpleGridWorld } from '../../src/RL/Environments/examples';

describe('Test Iterative Policy Evaluation', () => {
  it('should evaluate equiprobable policy on simple grid world correctly', () => {
    const width = 4;
    const height = 4;
    const targetPositions = [
      { x: 3, y: 3 },
      { x: 0, y: 0 },
    ];
    const env = new SimpleGridWorld(width, height, targetPositions, { x: 1, y: 0 });

    // the equiprobable policy
    const policy = () => {
      return 0.25;
    };
    const obsToStateRep = (state: ExtractStateType<SimpleGridWorld>) => {
      return state.agentPos.x + state.agentPos.y * Math.max(width, height);
    };
    const envFromStateRep = (stateString: string) => {
      const hash = parseInt(stateString);
      const m = Math.max(width, height);
      const x = hash % m;
      const y = Math.floor(hash / m);
      return new SimpleGridWorld(width, height, targetPositions, { x, y });
    };
    const allStateReps = [];
    for (let x = 0; x < env.width; x++) {
      for (let y = 0; y < env.height; y++) {
        const pos = { x: x, y: y };
        const env = new SimpleGridWorld(width, height, targetPositions, pos);
        allStateReps.push(obsToStateRep(env.reset()));
      }
    }
    const policyEvaluator = new DP.IterativePolicyEvaluation({
      obsToStateRep,
      envFromStateRep,
      allStateReps,
      policy,
      allPossibleActions: [0, 1, 2, 3],
    });
    policyEvaluator.train({
      verbose: false,
      steps: 10,
    });

    const gtVals = [
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
