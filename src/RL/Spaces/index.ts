import seedrandom from 'seedrandom';
import { prng } from '../utils/random';

export type Shape = number[];
export interface SpaceMeta {
  discrete: boolean;
}
/**
 * Abstract Space class. T is the type of data stored
 */
export abstract class Space<T> {
  // public static Box2D = Box2D;
  constructor(
    /** Meta data about the space such as if it is a discrete space or not */
    public meta: SpaceMeta
  ) {};
  public rng: prng = seedrandom();
  abstract sample(): T;
  abstract contains(x: T): boolean;
  seed(seed: string | undefined): void {
    this.rng = seedrandom(seed);
  }
}
export { Box } from './box';
export { PrimitiveBox2D } from './primitiveBox2d';
export { Discrete } from './discrete';
export { Dict } from './dict';
