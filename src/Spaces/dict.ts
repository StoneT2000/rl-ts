import { Space } from '.';

export class Dict<T extends Record<string, any>> extends Space<T> {
  constructor(public spaces: Record<string, Space<any>>) {
    super({ discrete: true });
    for (const k of Object.keys(this.spaces)) {
      if (!this.spaces[k].meta.discrete) {
        this.meta.discrete = false;
        break;
      }
    }
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
}
