/**
 * Async function that resolves after `ms` milliseconds
 * @param ms - number of milliseconds to sleep for
 */
export const sleep = async (ms: number): Promise<void> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, ms);
  });
};
