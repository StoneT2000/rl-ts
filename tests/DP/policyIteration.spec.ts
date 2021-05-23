import { expect } from 'chai';
import { DP } from '../../src';
import { IterativePolicyEvaluation } from '../../src/RL/DP/iterativePolicyEvaluation';
import { ExtractStateType } from '../../src/RL/Environments';
import { SimpleGridWorld } from '../../src/RL/Environments/examples';

describe('Test Policy Iteration', () => {
  it('should solve simple grid world', () => {
    let width = 4;
    let height = 4;
    let targetPositions = [
      { x: 3, y: 0 },
      { x: 0, y: 0 },
    ];
    let env = new SimpleGridWorld(width, height, targetPositions, { x: 1, y: 0 });

    let obsToStateRep = (obs: ExtractStateType<SimpleGridWorld>) => {
      return obs.agentPos.x + obs.agentPos.y * Math.max(width, height);
    };
    let envFromStateRep = (stateString: number) => {
      let hash = stateString;
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
        allStateReps.push(obsToStateRep(env.reset()));
      }
    }
    let policyIteration = new DP.PolicyIteration({
      obsToStateRep,
      envFromStateRep,
      allStateReps,
      allPossibleActions: [0, 1, 2, 3],
      discountRate: 0.5,
      evaluatorConfigs: {
        epsilon: 1e-1
      }
    });
    policyIteration.seed(0);
    policyIteration.train({
      untilStable: true,
      verbose: false,
    });

    // verify all actions produced are optimal
    allStateReps.forEach((stateRep) => {
      let obs = envFromStateRep(stateRep).reset();
      // optimal solution is closest target square
      let closestTargetPositions: any[] = [];
      let closestDist = 99;
      targetPositions.forEach((p) => {
        const dist = Math.abs(p.x - obs.agentPos.x) + Math.abs(p.y - obs.agentPos.y);
        if (dist <= closestDist) {
          closestDist = dist;
          closestTargetPositions.push(p)
        }
      });
      let optimalActions = [];
      if (closestDist == 0) return;
      for (const closestTargetPosition of closestTargetPositions) { 
        if (closestTargetPosition.y > obs.agentPos.y) {
          optimalActions.push(2);
        } else if (closestTargetPosition.y < obs.agentPos.y) {
          optimalActions.push(0);
        }
        if (closestTargetPosition.x > obs.agentPos.x) {
          optimalActions.push(1);
        } else if (closestTargetPosition.x < obs.agentPos.x) {
          optimalActions.push(3);
        }
      }
      expect(optimalActions).to.contain(policyIteration.action(obs))
    });
  });
});
