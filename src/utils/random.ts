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

export const randomRange = (
  rng: prng,
  low: number = Number.MIN_SAFE_INTEGER,
  high: number = Number.MAX_SAFE_INTEGER
): number => {
  return rng.double() * (high - low) + low;
};

/** return random value from [0, 1] */
export const randomVal = () => {
  return rng();
};

/** return array of given shape with values from [low, high] */
export const random = (shape: number[], low = 0, high = 1) => {
  const vals = nj.zeros(shape);
  for (let i = 0; i < vals.size; i++) {
    vals.selection.data[vals.selection.offset + i] = rng() * (high - low) + low;
  }
  return vals;
};
export const seed = (seed: number) => {
  rng = seedrandom(`${seed}`);
};
