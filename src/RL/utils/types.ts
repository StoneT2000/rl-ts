import ndarray from 'ndarray';
import nj from 'numjs';
import * as tf from '@tensorflow/tfjs';
/**
 * Makes every field in object T a partial type (optional)
 * @template T
 */
export type DeepPartial<T> = T extends object ? (T extends Function ? T : { [K in keyof T]?: DeepPartial<T[K]> }) : T;

declare global {
  /** A type that is any to avoid compilation errors for now but should be fixed in the future */
  type $TSFIXME = any;
  type TensorLike = tf.Tensor | nj.NdArray | number | number[] | number[][] | number[][][] | number[][][][];
}
declare module 'numjs' {
  interface NdArray<T> {
    selection: ndarray.NdArray;
    reshape(...shapes: number[]): NdArray<T>;
  }
}
