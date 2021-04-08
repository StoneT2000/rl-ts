import { Environment, RenderModes } from '..';
import { Box2D, Discrete } from '../../Spaces';

type State = number[][];
type Action = number;
type Position = { x: number; y: number };

/**
 * Simple GridWorld based on the gridworld presented in Chapters 3-4 in the Intro to RL book by Barto Sutton
 */
export class SimpleGridWorld extends Environment<Discrete, Box2D, Action, State, number> {
  // TODO: Expand gridworld to have other kinds of rewards and states (e.g. keys, chests, lava etc...)

  /** a 2D array with 0 representing a non-terminal tile/state and 1 representing a terminal tile, 2 representing position of agent */
  public observationSpace: Box2D;
  /** 0, 1, 2, 3 represent North, East, South, West directions */
  public actionSpace = new Discrete(4);

  public grid: State;
  public agentPos: Position;

  constructor(
    public width: number,
    public height: number,
    public targetPositions: Position[],
    public startPosition: Position
  ) {
    super();
    this.observationSpace = new Box2D(0, 1, [height, width]);
    this.grid = this.genGrid();
    this.agentPos = startPosition;
    this.grid[this.agentPos.y][this.agentPos.x] = 2;

    if (this.targetPositions.some((target) => !this.posOnGrid(target))) {
      throw new Error('One of the target positions are off the grid');
    }
    if (!this.posOnGrid(startPosition)) {
      throw new Error(`Start Position ${startPosition} is off the grid`);
    }
  }
  step(action: Action) {
    let reward = -1;
    let newPos = this.translate(this.agentPos, action);
    let done = false;
    if (this.posOnGrid(newPos)) {
      this.grid[this.agentPos.y][this.agentPos.x] = 0;
      this.agentPos = newPos;
      this.grid[this.agentPos.y][this.agentPos.x] = 2;
    }
    if (this.posIsInTargetPositions(this.agentPos)) {
      done = true;
      reward = 0;
    }
    return {
      observation: this.grid,
      reward,
      done,
      info: { width: this.width, height: this.height },
    };
  }
  private posIsInTargetPositions(pos: Position) {
    return this.targetPositions.some((target) => target.x == pos.x && target.y == pos.y);
  }
  private posOnGrid(pos: Position) {
    return !(pos.x < 0 || pos.y < 0 || pos.x >= this.width || pos.y >= this.height);
  }
  private translate(pos: Position, action: Action) {
    switch (action) {
      case 0:
        return { x: pos.x, y: pos.y - 1 };
      case 1:
        return { x: pos.x + 1, y: pos.y };
      case 2:
        return { x: pos.x, y: pos.y + 1 };
      case 3:
        return { x: pos.x - 1, y: pos.y };
      default:
        throw new Error(`Invalid action ${action}`);
    }
  }
  reset(): State {
    this.grid = this.genGrid();
    this.agentPos = this.startPosition;
    this.grid[this.agentPos.y][this.agentPos.x] = 2;
    return this.grid;
  }
  render(mode: RenderModes): void {
    if (mode === "human") {
      for (let y = 0; y < this.height; y++) {
        console.log(this.grid[y]);
      }
    }
  }
  private genGrid(): State {
    let grid = new Array(this.height);
    for (let y = 0; y < this.height; y++) {
      grid[y] = new Array(this.width);
      for (let x = 0; x < this.width; x++) {
        grid[y][x] = 0;
      }
    }
    this.targetPositions.forEach((pos) => {
      grid[pos.y][pos.x] = 1;
    });
    return grid;
  }
}
