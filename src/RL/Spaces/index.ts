import seedrandom from 'seedrandom';
import { prng } from '../utils/random';

export type Shape = number[];
/**
 * Abstract Space class. T is the type of data stored
 */
export abstract class Space<T> {
  // public static Box2D = Box2D;

  public rng: prng = seedrandom();
  abstract sample(): T;
  abstract contains(x: T): boolean;
  abstract to_jsonable(sample_n: T[]): any;
  abstract from_jsonable(sample_n: T[]): any;
  seed(seed: string | undefined): void {
    this.rng = seedrandom(seed);
  }
}
export { PrimitiveBox2D } from './primitiveBox2d';
export { PrimitiveDiscrete } from './primitiveDiscrete';
export { Dict } from './dict';
