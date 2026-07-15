interface QueueResult<T> {
  value?: T;
  done: boolean;
}

export class AsyncQueue<T> {
  private readonly values: T[] = [];
  // Index of the next value to consume from `values`. We pop from the head
  // without re-indexing the array (which `Array.prototype.shift` would do in
  // O(n)), so the amortised cost of draining is O(1) per item instead of O(n).
  private head = 0;
  private readonly waiters: Array<{
    resolve: (result: QueueResult<T>) => void;
    reject: (error: unknown) => void;
  }> = [];

  private finished = false;
  private failure: unknown;

  push(value: T): void {
    if (this.finished) {
      return;
    }

    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve({ value, done: false });
      return;
    }

    this.values.push(value);
  }

  /**
   * Reclaims the leading consumed slice of `values` once in a while to keep
   * the backing array from growing without bound during long-running streams.
   * Called only on the slow path (no buffered values), so the occasional
   * O(n) compaction is amortised over many push/next pairs.
   */
  private compact(): void {
    if (this.head > 0) {
      if (this.head < this.values.length) {
        this.values.copyWithin(0, this.head, this.values.length);
        this.values.length = this.values.length - this.head;
      } else {
        this.values.length = 0;
      }
      this.head = 0;
    }
  }

  end(): void {
    if (this.finished) {
      return;
    }

    this.finished = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.resolve({ done: true });
    }
  }

  fail(error: unknown): void {
    if (this.finished) {
      return;
    }

    this.finished = true;
    this.failure = error;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }

  isFinished(): boolean {
    return this.finished;
  }

  async next(): Promise<QueueResult<T>> {
    if (this.head < this.values.length) {
      const value = this.values[this.head];
      this.values[this.head] = undefined as unknown as T;
      this.head += 1;
      return { value, done: false };
    }

    if (this.failure !== undefined) {
      throw this.failure;
    }

    if (this.finished) {
      return { done: true };
    }

    // Nothing buffered: compact the array so the backing store doesn't keep
    // growing while we wait for producers, then park on a waiter.
    this.compact();

    return new Promise<QueueResult<T>>((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }
}
