/** File contains implementations of basic numpy operations using ndarray */
import { Tensor, tensor } from '@tensorflow/tfjs-core';
import ndarray, { DataType } from 'ndarray';
import ops from 'ndarray-ops';
// eslint-disable-next-line
//@ts-ignore
import _pack from 'ndarray-pack';
// eslint-disable-next-line
//@ts-ignore
import _unpack from 'ndarray-unpack';
import nj, { NdArray } from 'numjs';
import { NotImplementedError } from 'rl-ts/lib/Errors';

export const pack = (arr: Array<any>, dtype?: DataType): NdArray<any> => {
  return nj.array(arr, dtype);
};
export const packNp = (arr: Array<any>): ndarray.NdArray<any> => {
  return _pack(arr);
};
export const unpack = (arr: NdArray<any>): Array<any> => {
  return arr.tolist();
};
export const unpackNp = (arr: ndarray.NdArray<any>): Array<any> => {
  return _unpack(arr);
};

export const types = {
  float32: Float32Array,
  float64: Float64Array,
  int64: Int16Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
  string: Array,
};
export type dtype = keyof typeof types;

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
export const zeros = (shape: number[], dtype: dtype = 'float32') => {
  return ndarray(new types[dtype](reduceMult(shape)), shape);
};

export const tensorLikeToNdArray = (x: TensorLike): NdArray => {
  if (x instanceof Tensor) {
    x = fromTensorSync(x);
  } else if (x instanceof Array) {
    x = pack(x);
  } else if (typeof x === 'number') {
    x = nj.array([x]);
  }
  return x;
};
export const tensorLikeToTensor = (x: TensorLike): Tensor => {
  if (x instanceof Tensor) {
    return x;
  } else if (x instanceof Array) {
    x = pack(x);
  } else if (typeof x === 'number') {
    x = nj.array([x]);
  }
  return tensor(x.selection.data).reshape(x.shape);
};
export const NdArrayToTensorLike = (x: NdArray, target: TensorLike): TensorLike => {
  if (target instanceof Tensor) {
    return tensor(x.selection.data).reshape(x.shape);
  } else if (target instanceof Array) {
    return unpack(x);
  } else if (typeof target === 'number') {
    return x.get(0);
  }
  return x;
};

/**
 * Converts tensor to NdArray
 * @param tensor
 */
export const fromTensor = async (tensor: Tensor) => {
  const data = await tensor.data();
  return nj.array(data, tensor.dtype as ndarray.DataType);
};

/**
 * Converts tensor to numjs NdArray synchronously
 * @param tensor
 */
export const fromTensorSync = (tensor: Tensor) => {
  const data = tensor.arraySync();
  return nj.array(data as number[], tensor.dtype as ndarray.DataType);
};

/**
 * Converts tensor to NdArray synchronously
 * @param tensor
 */
export const fromTensorSyncToNp = (tensor: Tensor): ndarray.NdArray => {
  const data = tensor.dataSync();
  return ndarray(data, tensor.shape);
};

export const toTensor = (x: NdArray) => {
  return tensor(unpack(x));
};

export const toTensorFromNp = (x: ndarray.NdArray) => {
  return tensor(_unpack(x));
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
  if (typeof val === 'number') {
    if (arr.shape.length !== 1) {
      throw new Error('Array is not a 1-dimensional vector');
    }
    return nj.array([...unpack(arr), val]);
  } else {
    throw new NotImplementedError('TODO');
  }
};

/**
 * A better set function that allows setting other NdArrays in others in place or using a boolean mask
 *
 * @param arr
 */
export const set = (
  src: ndarray.NdArray,
  index: number[] | ndarray.NdArray<any>,
  val: ndarray.NdArray<any> | number
): ndarray.NdArray => {
  // if index argument is a NdArray, expect it to be a boolean mask.
  if (index instanceof Array) {
    // verify shapes are valid
    if (index.length > src.shape.length)
      throw new Error(
        `Indexing tried to index ${index.length} dimensions but src array has only ${src.shape.length} dimensions`
      );
    if (index.length === src.shape.length) {
      // if indexing a single value, allow any value to be set provides it fits in dimensions
      let scalar: number;
      if (typeof val !== 'number') {
        if (!arrayEqual(val.shape, [1])) {
          throw new Error('Only size-1 arrays can be converted to a scalar');
        }
        scalar = val.get(0);
      } else {
        scalar = val;
      }
      src.set(...index, scalar);
      return src;
    } else {
      if (typeof val === 'number') {
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
  } else {
    // find every element that is a 1 (True)
    if (!arrayEqual(src.shape, index.shape)) {
      throw new Error(`Boolean mask shape ${index.shape} mismatched with ${src.shape}`);
    }

    const positiveIndices: number[] = [];
    for (let i = 0; i < reduceMult(index.shape); i++) {
      const v = loc(index, i);
      if (v === 1) {
        positiveIndices.push(i);
      }
    }

    if (typeof val === 'number') {
      for (let i = 0; i < positiveIndices.length; i++) {
        src.data[positiveIndices[i]] = val;
      }
    } else {
      if (positiveIndices.length === 0) return src;
      if (reduceMult(val.shape) !== positiveIndices.length) {
        throw new Error(
          `Cannot assign ${val.shape} input values to ${positiveIndices.length} output values when masking`
        );
      }
      for (let i = 0; i < positiveIndices.length; i++) {
        src.data[positiveIndices[i]] = loc(val, i);
      }
    }
    return src;
  }
};
/** Gets ith value */
export const loc = (x: ndarray.NdArray, i: number) => {
  return x.data[x.offset + i];
};

export const fromNj = (x: NdArray): ndarray.NdArray => {
  return _pack(x.tolist());
};
export const toNj = (x: ndarray.NdArray) => {
  return nj.array(_unpack(x), x.dtype);
};

// this exists because the package's reshape function does not quite work
export const unsqueeze = (x: NdArray, index: number, copy = true): NdArray => {
  if (copy) {
    x = x.clone();
  }
  x.selection.shape = [...x.shape.slice(0, index), 1, ...x.shape.slice(index)];

  // compute the new stride
  let shapemult = 1;
  x.selection.stride = new Array(x.shape.length);
  for (let i = x.shape.length - 1; i >= 0; i--) {
    shapemult = shapemult * x.shape[i];
    x.selection.stride[i] = Math.floor(shapemult / x.shape[i]);
  }
  return x;
};
