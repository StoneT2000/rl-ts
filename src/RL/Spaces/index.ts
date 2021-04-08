import seedrandom from 'seedrandom';
import { prng } from '../utils/random';

export type Shape = number[];
/**
 * Abstract Space class. T is the type of data stored
 */
export abstract class Space<T> {
  // public static Box2D = Box2D;

  public rng: prng = seedrandom();
  constructor() {}
  abstract sample(): T;
  abstract contains(x: T): boolean;
  abstract to_jsonable(sample_n: T[]): any;
  abstract from_jsonable(sample_n: T[]): any;
  seed(seed: string | undefined): void {
    this.rng = seedrandom(seed);
  }
}
export { Box2D } from './box';
export { Discrete } from './discrete';
export {Dict} from './dict';