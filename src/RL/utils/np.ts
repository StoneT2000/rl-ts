/** File contains implementations of basic numpy operations using ndarray */
import { Tensor } from '@tensorflow/tfjs-core';
import ndarray, { NdArray } from 'ndarray';
import ops from 'ndarray-ops';
// eslint-disable-next-line
//@ts-ignore
import _pack from 'ndarray-pack';

export const pack: (arr: Array<any>) => NdArray<any> = _pack;

export const types = {
  float32: Float32Array,
  float64: Float64Array,
  int64: Int16Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
  string: Array,
};
type DType = keyof typeof types;

export const reduceMult = (arr: number[] | ndarray.NdArray): number => {
  let m: number[];
  if (arr instanceof Array) {
    m = arr;
  } else {
    m = arr.data as number[];
  }
  let ret = 1;
  for (const v of m) {
    ret *= v;
  }
  return ret;
};

export const zeros = (shape: number[], dtype: DType = 'float32') => {
  return ndarray(new types[dtype](reduceMult(shape)), shape);
};

/**
 * Converts tensor to NdArray
 * @param tensor
 */
export const fromTensor = async (tensor: Tensor) => {
  const data = await tensor.data();
  return ndarray(data, tensor.shape);
};

export const arrayEqual = <T>(arr1: T[], arr2: T[]): boolean => {
  if (arr1.length !== arr2.length) return false;
  for (let i = 0; i < arr1.length; i++) {
    if (arr1[i] !== arr2[i]) return false;
  }
  return true;
};

/**
 * A better set function that allows setting other NdArrays
 *
 * Not in place
 *
 * @param arr
 */
export const set = (src: NdArray, index: number[], val: NdArray<any>): NdArray => {
  // verify shapes are valid
  if (index.length > src.shape.length)
    throw new Error(
      `Indexing tried to index ${index.length} dimensions but src array has only ${src.shape.length} dimensions`
    );
  if (index.length === src.shape.length) {
    // if indexing a single value, allow any value to be set provides it fits in dimensions
    if (!arrayEqual(val.shape, [1])) {
      throw new Error('Only size-1 arrays can be converted to a scalar');
    } else {
      src.set(...index, val.get(0));
      return src;
    }
  } else {
    const requiredShape = src.shape.slice(index.length);

    if (reduceMult(val.shape) !== reduceMult(requiredShape)) {
      throw new Error(`Cannot broadcast input array from shape ${val.shape} to ${requiredShape}`);
    }

    for (let i = 0; i < index.length; i++) {
      const d = index[i];
      if (d >= src.shape[i]) {
        throw new Error(`Index ${d} out of bounds in ${i}th dimension of source array with shape ${src.shape}`);
      }
    }

    const subDest = src.pick(...index);
    ops.assign(subDest, val);
    return src;
  }
};

const a = zeros([3, 2, 2]);
console.log(a);
// let b = ndarray([[20, 30], [3,4]],[2,2])
const b = pack([
  [20, 3],
  [4, 2],
]);
// console.log(b.shape)
const c = set(a, [0], b);
console.log(a, c);
