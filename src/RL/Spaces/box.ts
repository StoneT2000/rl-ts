import { DataType, NdArray } from 'ndarray';
import { Shape, Space } from '.';
import * as random from '../utils/random';
import * as np from '../utils/np';
import ops from 'ndarray-ops';
import * as tf from '@tensorflow/tfjs-node';
import { NotImplementedError } from '../Errors';
import nj from 'numjs';

export class Box extends Space<nj.NdArray<number>> {

  public low: null | NdArray<number>;
  public high: null | NdArray<number>;

  public boundedBelow:  NdArray<number>;//tf.Tensor;
  public boundedAbove:  NdArray<number>;//tf.Tensor;

  /**
   * @param low - lower bound for all dimensions or a list of numbers representing lower bounds for each dimension of given shape
   * @param high - upper bound for all dimensions or a list of numbers representing upper bounds for each dimension of given shape
   * @param shape - the desired box shape
   */
  constructor(low: null | number | nj.NdArray<number>, high: null | number | nj.NdArray<number>, public shape: Shape, public dtype: DataType = "float32") {
    super();
    let lowerBounds: null | nj.NdArray<number> = nj.zeros(shape, dtype);
    let upperBounds: null | nj.NdArray<number> = nj.zeros(shape, dtype);
    if (typeof low === "number") {
      lowerBounds.assign(low, false);
    } else if (low === null) {
      lowerBounds = null;
    } else {
      if (!np.arrayEqual(lowerBounds.shape, low.shape)) throw new Error(`low shape ${low.shape} does not match given shape ${shape}`);
      lowerBounds = low;
    }
    if (typeof high === "number") {
      upperBounds.assign(high, false);
    } else if (high === null) {
      upperBounds = null;
    } else {
      if (!np.arrayEqual(upperBounds.shape, high.shape)) throw new Error(`high shape ${high.shape} does not match given shape ${shape}`);
      upperBounds = high;
    }

    this.low = lowerBounds ? np.fromNj(lowerBounds) : lowerBounds;
    this.high = upperBounds ? np.fromNj(upperBounds) : upperBounds;

    this.boundedBelow = np.zeros(shape);
    this.boundedAbove = np.zeros(shape);
    
    if (this.low) {
      this.boundedBelow = ops.gts(this.boundedBelow, this.low, Number.MIN_SAFE_INTEGER);
      // this.boundedBelow = tf.less(this.boundedBelow, Number.MIN_SAFE_INTEGER);
    }
    if (this.high) {
      // this.boundedAbove = tf.greater(this.boundedAbove, Number.MAX_SAFE_INTEGER);
      this.boundedAbove = ops.lts(this.boundedAbove, this.high, Number.MAX_SAFE_INTEGER);
    }
  }
  sample(): nj.NdArray<number> {
    let sample = np.zeros(this.shape);
    // determine which indices are lower bounded and upper bounded
    if (this.boundedBelow === null && this.boundedAbove === null) {
      return np.fromTensorSync(tf.randomNormal(this.shape));
    }

    let boundedBelowNeg = np.zeros(this.shape, "float32");
    let boundedAboveNeg = np.zeros(this.shape, "float32");

    ops.neg(boundedBelowNeg, this.boundedBelow);
    ops.addseq(boundedBelowNeg, 1);
    ops.neg(boundedAboveNeg, this.boundedAbove);
    ops.addseq(boundedAboveNeg, 1);

    let unbounded = np.zeros(this.shape, "int8");
    let lowBounded = np.zeros(this.shape, "int8");
    let upperBounded = np.zeros(this.shape, "int8");
    let bounded = np.zeros(this.shape, "int8");
    
    ops.and(unbounded, boundedBelowNeg, boundedAboveNeg);
    ops.and(lowBounded, this.boundedBelow, boundedAboveNeg);
    ops.and(upperBounded, this.boundedAbove, boundedBelowNeg);
    ops.and(bounded, this.boundedBelow, this.boundedAbove);

    np.set(sample, unbounded, np.fromTensorSync(tf.randomNormal(this.shape)));

    if (this.low) {
      
      let val = np.fromTensorSyncToNp(tf.randomGamma(this.shape, 1, 1).add(np.toTensorFromNp(this.low)));
      np.set(sample, lowBounded, val);
    }
    if (this.high) {
      np.set(sample, upperBounded, np.fromTensorSyncToNp(tf.randomGamma(this.shape, 1, 1).neg().add(np.toTensorFromNp(this.high))));
    }
    if (this.low && this.high) {
      // generate uniform with shape this.shape
      let vals = np.zeros(this.shape);
      for (let i = 0; i < np.reduceMult(this.low.shape); i++ ) {
        let l = np.loc(this.low, i);
        let h = np.loc(this.high, i);
        vals.data[i] = (random.random() as number) * (h - l) + l;
      }
      np.set(sample, bounded, vals);
    }
    return np.toNj(sample);
  }
  contains(x: NdArray): boolean {
    // // verify shape
    // if (x.length !== this.shape[0] || x[0].length !== this.shape[1]) {
    //   return false;
    // }
    // // verify contents
    // for (let i = 0; i < this.shape[0]; i++) {
    //   for (let j = 0; j < this.shape[1]; j++) {
    //     if (x[i][j] < this.low || x[i][j] > this.high) {
    //       return false;
    //     }
    //   }
    // }
    throw new NotImplementedError("Not implemented yet");
    return true;
  }
  to_jsonable(sample_n: NdArray[]) {
    return sample_n;
  }
  from_jsonable(sample_n: NdArray[]) {
    return sample_n;
  }
}
