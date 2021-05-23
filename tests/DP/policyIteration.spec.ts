import { expect } from 'chai';
import { DP } from '../../src';
import { ExtractStateType } from '../../src/RL/Environments';
import { SimpleGridWorld } from '../../src/RL/Environments/examples';

describe('Test Policy Iteration', () => {
  it('should solve simple grid world', () => {
    const width = 4;
    const height = 4;
    const targetPositions = [
      { x: 3, y: 0 },
      { x: 0, y: 0 },
    ];
    const env = new SimpleGridWorld(width, height, targetPositions, { x: 1, y: 0 });

    const obsToStateRep = (obs: ExtractStateType<SimpleGridWorld>) => {
      return obs.agentPos.x + obs.agentPos.y * Math.max(width, height);
    };
    const envFromStateRep = (stateString: number) => {
      const hash = stateString;
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
    const policyIteration = new DP.PolicyIteration({
      obsToStateRep,
      envFromStateRep,
      allStateReps,
      allPossibleActions: [0, 1, 2, 3],
      discountRate: 0.5,
      evaluatorConfigs: {
        epsilon: 1e-1,
      },
    });
    policyIteration.seed(0);
    policyIteration.train({
      untilStable: true,
      verbose: false,
    });

    // verify all actions produced are optimal
    allStateReps.forEach((stateRep) => {
      const obs = envFromStateRep(stateRep).reset();
      // optimal solution is closest target square
      const closestTargetPositions: any[] = [];
      let closestDist = 99;
      targetPositions.forEach((p) => {
        const dist = Math.abs(p.x - obs.agentPos.x) + Math.abs(p.y - obs.agentPos.y);
        if (dist <= closestDist) {
          closestDist = dist;
          closestTargetPositions.push(p);
        }
      });
      const optimalActions = [];
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
      expect(optimalActions).to.contain(policyIteration.action(obs));
    });
  });
});
