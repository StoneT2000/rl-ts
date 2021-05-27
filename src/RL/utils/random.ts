import seedrandom from 'seedrandom';
import nj from 'numjs';
/** global rng */
let rng = seedrandom(`${Math.random()}`);

export interface prng {
  (): number;
  double(): number;
  int32(): number;
  quick(): number;
  state(): seedrandom.State;
}

export const randomRange = (rng: prng, low: number = Number.MIN_SAFE_INTEGER, high: number = Number.MAX_SAFE_INTEGER): number => {
  return rng.double() * (high - low) + low;
};

/** return random value from [0, 1] */
export const random = (shape?: number[]) => {
  if (shape === undefined) {
    return rng();
  }
  let vals = nj.zeros(shape);
  for (let i = 0; i < vals.data.length; i++) {
    vals.data[vals.offset + i] = rng();
  };
  return vals;
};

export const seed = (seed: number) => {
  rng = seedrandom(`${seed}`);
}