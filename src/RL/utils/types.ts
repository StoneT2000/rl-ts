import seedrandom from "seedrandom";

interface prng {
  (): number;
  double(): number;
  int32(): number;
  quick(): number;
  state(): seedrandom.State;
}