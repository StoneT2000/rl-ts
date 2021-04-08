import { Environment, RenderModes } from '..';
import { Box2D, Dict, Discrete } from '../../Spaces';

type State = { grid: number[][]; agentPos: Position };
type Action = number;
type Position = { x: number; y: number };
type ObservationSpace = Dict<State>;

/**
 * Simple GridWorld based on the gridworld presented in Chapters 3-4 in the Intro to RL book by Barto Sutton
 */
export class SimpleGridWorld extends Environment<Discrete, ObservationSpace, Action, State, number> {
  // TODO: Expand gridworld to have other kinds of rewards and states (e.g. keys, chests, lava etc...)

  /** a 2D array with 0 representing a non-terminal tile/state and 1 representing a terminal tile, 2 representing position of agent */
  public observationSpace: ObservationSpace;
  /** 0, 1, 2, 3 represent North, East, South, West directions */
  public actionSpace = new Discrete(4);

  public state: State;

  constructor(
    public width: number,
    public height: number,
    public targetPositions: Position[],
    public startPosition: Position
  ) {
    super();
    this.observationSpace = new Dict({
      grid: new Box2D(0, 1, [2, 2]),
      agentPos: new Dict({
        x: new Discrete(4),
        y: new Discrete(4),
      }),
    });
    this.state = this.genState();

    if (this.targetPositions.some((target) => !this.posOnGrid(target))) {
      throw new Error('One of the target positions are off the grid');
    }
    if (!this.posOnGrid(startPosition)) {
      throw new Error(`Start Position ${startPosition} is off the grid`);
    }
  }
  step(action: Action) {
    let reward = -1;
    let newPos = this.translate(this.state.agentPos, action);
    let done = false;
    if (this.posOnGrid(newPos)) {
      this.state.agentPos = newPos;
    }
    if (this.posIsInTargetPositions(this.state.agentPos)) {
      done = true;
    }
    return {
      observation: this.get_obs(),
      reward,
      done,
      info: { width: this.width, height: this.height },
    };
  }
  posIsInTargetPositions(pos: Position) {
    return this.targetPositions.some((target) => target.x == pos.x && target.y == pos.y);
  }
  posOnGrid(pos: Position) {
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
    this.state = this.genState();
    return this.get_obs();
  }
  private get_obs(): State {
    return JSON.parse(JSON.stringify(this.state));
  }
  render(mode: RenderModes): void {
    let obs = this.get_obs();
    obs.grid[obs.agentPos.y][obs.agentPos.x] = 2;
    if (mode === 'human') {
      for (let y = 0; y < this.height; y++) {
        console.log(obs.grid[y]);
      }
    }
  }
  private genState(): State {
    let grid = this.genGrid();
    let agentPos = this.startPosition;
    return { grid, agentPos };
  }
  private genGrid(): number[][] {
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
