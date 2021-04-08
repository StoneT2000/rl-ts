import { Shape, Space } from ".";
import { randomRange } from "../utils/random";

export class Box2D extends Space<number[][]> {
  constructor(public low: number, public high: number, shape: Shape) {
    super(shape);
    if (shape.length != 2) {
      throw new Error("Shape must be 2D");
    }
  }
  sample(): number[][] {
    let sample: number[][] = new Array(this.shape[0]);
    for (let i = 0; i < this.shape[0]; i++) {
      sample[i] = new Array(this.shape[1]);
      for (let j = 0; j < this.shape[1]; j++) {
        sample[i][j] = randomRange(this.rng, this.low, this.high);
      }
    }
    return sample;
  }
  contains(x: number[][]): boolean {
    throw new Error("Method not implemented.");
  }
  to_jsonable(sample_n: number[][][]) {
    throw new Error("Method not implemented.");
  }
  from_jsonable(sample_n: number[][][]) {
    throw new Error("Method not implemented.");
  }

}