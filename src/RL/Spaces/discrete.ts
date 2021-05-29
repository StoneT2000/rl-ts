import { Space } from '.';
import { randomRange } from '../utils/random';

/**
 * Primitive discrete space with values from set {0, 1, ..., n - 1}.
 */
export class Discrete extends Space<number> {
  constructor(public n: number) {
    super({ discrete: true });
    this.shape = [1];
  }
  sample(): number {
    return Math.floor(randomRange(this.rng, 0, this.n));
  }
  contains(x: number): boolean {
    return x < this.n && x >= 0;
  }
}
