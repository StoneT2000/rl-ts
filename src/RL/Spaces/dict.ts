import { Space } from '.';

export class Dict<T extends Record<string, any>> extends Space<T> {
  constructor(public spaces: Record<string, Space<any>>) {
    super();
  }
  sample() {
    const sample: Record<string, any> = {};
    for (const k of Object.keys(this.spaces)) {
      sample[k] = this.spaces[k].sample();
    }
    return sample as T;
  }
  contains(x: T): boolean {
    for (const k of Object.keys(this.spaces)) {
      if (!this.spaces[k].contains(x[k])) return false;
    }
    return true;
  }
  to_jsonable(sample_n: T[]) {
    return sample_n;
  }
  from_jsonable(sample_n: T[]) {
    return sample_n;
  }
}
