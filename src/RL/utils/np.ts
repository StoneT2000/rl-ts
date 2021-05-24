/** File contains implementations of basic numpy operations using ndarray */
import { Tensor } from '@tensorflow/tfjs-core';
import ndarray, { NdArray } from 'ndarray';
import ops from 'ndarray-ops';
// eslint-disable-next-line
//@ts-ignore
import _pack from 'ndarray-pack';
// eslint-disable-next-line
//@ts-ignore
import _unpack from 'ndarray-unpack';
import { NotImplementedError } from '../Errors';

export const pack: (arr: Array<any>) => NdArray<any> = _pack;
export const unpack: (arr: NdArray<any>) => Array<any> = _unpack;

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
 * Copies arr and pushes val to the end of it and returns the copy
 * @param arr
 * @param val 
 */
export const push = (arr: NdArray<any>, val: number | NdArray): NdArray => {
  if (typeof val === "number") {
    if (arr.shape.length !== 1) {
      throw new Error("Array is not a 1-dimensional vector");
    }
    return ndarray([...unpack(arr), val], [arr.shape[0] + 1]);
  }
  else {
    throw new NotImplementedError("TODO");
  }
}

/**
 * A better set function that allows setting other NdArrays
 *
 * Not in place
 *
 * @param arr
 */
export const set = (src: NdArray, index: number[], val: NdArray<any> | number): NdArray => {
  // verify shapes are valid
  if (index.length > src.shape.length)
    throw new Error(
      `Indexing tried to index ${index.length} dimensions but src array has only ${src.shape.length} dimensions`
    );
  if (index.length === src.shape.length) {
    // if indexing a single value, allow any value to be set provides it fits in dimensions
    let scalar: number;
    if (typeof val !== "number") {
      if (!arrayEqual(val.shape, [1])) {
        throw new Error('Only size-1 arrays can be converted to a scalar');
      };
      scalar = val.get(0);
    }
    else {
      scalar = val;
    }
    src.set(...index, scalar);
    return src;
  } else {
    if (typeof val === "number") {
      throw new Error(`Cannot set a scalar value`);
    }
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