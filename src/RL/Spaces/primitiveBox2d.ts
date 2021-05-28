import { Shape, Space } from '.';
import { randomRange } from '../utils/random';

/**
 * A Box 2d space. Uses normal JS numbers and not ndarrays
 */
export class PrimitiveBox2D extends Space<number[][]> {
  constructor(public low: number, public high: number, public shape: Shape) {
    super({discrete: false});
    if (shape.length !== 2) {
      throw new Error('Shape must be 2D');
    }
  }
  sample(): number[][] {
    const sample: number[][] = new Array(this.shape[0]);
    for (let i = 0; i < this.shape[0]; i++) {
      sample[i] = new Array(this.shape[1]);
      for (let j = 0; j < this.shape[1]; j++) {
        sample[i][j] = randomRange(this.rng, this.low, this.high);
      }
    }
    return sample;
  }
  contains(x: number[][]): boolean {
    // verify shape
    if (x.length !== this.shape[0] || x[0].length !== this.shape[1]) {
      return false;
    }
    // verify contents
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        if (x[i][j] < this.low || x[i][j] > this.high) {
          return false;
        }
      }
    }
    return true;
  }
  to_jsonable(sample_n: number[][][]) {
    return sample_n;
  }
  from_jsonable(sample_n: number[][][]) {
    return sample_n;
  }
}
