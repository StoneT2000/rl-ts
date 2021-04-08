import seedrandom from "seedrandom";

export interface prng {
  (): number;
  double(): number;
  int32(): number;
  quick(): number;
  state(): seedrandom.State;
}

export const randomRange = (rng: prng, low: number, high: number) => {
  return rng.double() * (high - low)  + low;
}