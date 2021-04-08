import * as RL from "../src";
let space = new RL.Spaces.Box2D(0, 1, [2, 2]);
console.log(space.sample());