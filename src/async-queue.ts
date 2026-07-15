interface QueueResult<T> {
  value?: T;
  done: boolean;
}

export class AsyncQueue<T> {
  private readonly values: T[] = [];
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
    const value = this.values.shift();
    if (value !== undefined) {
      return { value, done: false };
    }

    if (this.failure !== undefined) {
      throw this.failure;
    }

    if (this.finished) {
      return { done: true };
    }

    return new Promise<QueueResult<T>>((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }
}
