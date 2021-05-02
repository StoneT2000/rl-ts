/**
 * Error that is thrown when something is not implemented
 */
 export class NotImplementedError extends Error {
  constructor(m: string) {
    super(m);
    this.name = "Not implemented";
    Object.setPrototypeOf(this, NotImplementedError.prototype);
  }
}