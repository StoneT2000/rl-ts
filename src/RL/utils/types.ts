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
  /** Save locations supported. See https://www.tensorflow.org/js/guide/save_load#save_a_tfmodel for what each of options do */
  type TFSaveLocations = 'localstorage' | 'indexeddb' | 'downloads' | 'file';
}
declare module 'numjs' {
  interface NdArray<T> {
    selection: ndarray.NdArray;
    reshape(...shapes: number[]): NdArray<T>;
  }
}
