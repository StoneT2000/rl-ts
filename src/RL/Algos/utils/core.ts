import nj, { NdArray } from 'numjs';

/**
 * Computes [x0 + discount * x1 + discount^2 * x2 + ... + discount^n * xn, x1 + discount * x2 + ..., ..., xn-2 + discount * xn-1, xn-1] in O(n) time and space
 * @param x
 * @param discount
 */
export const discountCumSum = (x: NdArray, discount: number): NdArray => {
  const n = x.shape[0];
  const cumsum = nj.zeros(x.shape);
  cumsum.set(n - 1, x.get(n - 1));
  for (let i = 1; i < n; i++) {
    const prev = cumsum.get(n - i);
    cumsum.set(n - i - 1, x.get(n - i - 1) + prev * discount);
  }
  return cumsum;
};
