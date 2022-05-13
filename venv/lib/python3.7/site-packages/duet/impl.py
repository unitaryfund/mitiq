# Copyright 2021 The Duet Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal implementation details for duet."""

import enum
import functools
import heapq
import itertools
import signal
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar
from typing import (
    Any,
    Awaitable,
    Callable,
    cast,
    Coroutine,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
)

import duet.futuretools as futuretools

T = TypeVar("T")


class Interrupt(BaseException):
    def __init__(self, task, error):
        self.task = task
        self.error = error


class TaskState(enum.Enum):
    WAITING = 0
    SUCCEEDED = 1
    FAILED = 2


class TaskStateError(Exception):
    def __init__(self, state: TaskState, expected_state: TaskState) -> None:
        self.state = state
        self.expected_state = expected_state
        super().__init__(f"state: {state}, expected: {expected_state}")


# Sentinel local variable name that we insert into coroutines.
# This allows us to detect whether a task is running when we get Ctrl-C.
LOCALS_TASK_SCHEDULER = "__duet_task_scheduler__"


class Task(Generic[T]):
    def __init__(
        self, awaitable: Awaitable[T], scheduler: "Scheduler", main_task: Optional["Task"]
    ) -> None:
        self.scheduler = scheduler
        self.main_task = main_task
        self._state = TaskState.WAITING
        self._future: Optional[Future] = None
        self._ready_future = futuretools.AwaitableFuture[None]()
        self._ready_future.set_result(None)  # Ready to advance.
        self.interruptible = True
        self._interrupt: Optional[Interrupt] = None
        self._result: Optional[T] = None
        self._error: Optional[Exception] = None
        self._deadlines: List[DeadlineEntry] = []
        if main_task and main_task.deadline_entry is not None:
            entry = main_task.deadline_entry
            self.push_deadline(deadline=entry.deadline, timeout_error=entry.timeout_error)
        self._generator = awaitable.__await__()  # Returns coroutine generator.
        if isinstance(awaitable, Coroutine):
            awaitable.cr_frame.f_locals.setdefault(LOCALS_TASK_SCHEDULER, scheduler)

    def _check_state(self, expected_state: TaskState) -> None:
        if self._state != expected_state:
            raise TaskStateError(self._state, expected_state)

    @property
    def future(self) -> Optional[Future]:
        self._check_state(TaskState.WAITING)
        return self._future

    @property
    def result(self) -> T:
        self._check_state(TaskState.SUCCEEDED)
        return cast(T, self._result)

    @property
    def done(self) -> bool:
        return self._state == TaskState.SUCCEEDED or self._state == TaskState.FAILED

    def add_ready_callback(self, callback: Callable[["Task"], Any]) -> None:
        self._check_state(TaskState.WAITING)
        self._ready_future.add_done_callback(lambda _: callback(self))

    def advance(self):
        if self.done:
            return
        if self._state == TaskState.WAITING:
            self._ready_future.result()
        token = _current_task.set(self)
        try:
            if self._interrupt:
                interrupt = self._interrupt
                self._interrupt = None
                if interrupt.task is self:
                    error = interrupt.error
                else:
                    error = interrupt
                f = self._generator.throw(error)
            else:
                f = next(self._generator)
        except StopIteration as e:
            self._result = e.value
            self._state = TaskState.SUCCEEDED
            return
        except (Interrupt, Exception) as error:
            self._error = error
            self._state = TaskState.FAILED
            if self.main_task:
                self.main_task.interrupt(self, error)
                return
            else:
                raise
        else:
            if not isinstance(f, Future):
                raise TypeError(f"expected Future, got {type(f)}: {f}")
            ready_future = futuretools.AwaitableFuture()
            f.add_done_callback(lambda _: ready_future.try_set_result(None))
            self._future = f
            self._ready_future = ready_future
            self._state = TaskState.WAITING
        finally:
            _current_task.reset(token)

    def push_deadline(self, deadline: float, timeout_error: TimeoutError) -> None:
        if self._deadlines:
            entry = self._deadlines[-1]
            if entry.deadline < deadline:
                deadline = entry.deadline
                timeout_error = entry.timeout_error
        entry = DeadlineEntry(self, deadline, timeout_error)
        self.scheduler.add_deadline(entry)
        self._deadlines.append(entry)

    def pop_deadline(self) -> None:
        entry = self._deadlines.pop(-1)
        entry.valid = False

    @property
    def deadline_entry(self) -> Optional["DeadlineEntry"]:
        return self._deadlines[-1] if self._deadlines else None

    def interrupt(self, task, error):
        if self.done or not self.interruptible or self._interrupt:
            return
        self._interrupt = Interrupt(task, error)
        self._ready_future.try_set_result(None)
        if self._future:
            self._future.cancel()

    def close(self):
        self._generator.close()
        self.scheduler = None
        self.main_task = None


_current_task: ContextVar[Task] = ContextVar("current_task")


def current_task() -> Task:
    """Gets the currently-running duet task.

    This must be called from within a running async function, or else it will
    raise a RuntimeError.
    """
    try:
        return _current_task.get()
    except LookupError:
        raise RuntimeError("Can only be called from an async function.")


def current_scheduler() -> "Scheduler":
    """Gets the currently-running duet scheduler.

    This must be called from within a running async function, or else it will
    raise a RuntimeError.
    """
    return current_task().scheduler


def any_ready(tasks: Set[Task]) -> futuretools.AwaitableFuture[None]:
    """Returns a Future that will fire when any of the given tasks is ready."""
    if not tasks or any(task.done for task in tasks):
        return futuretools.completed_future(None)
    f = futuretools.AwaitableFuture[None]()
    for task in tasks:
        task.add_ready_callback(lambda _: f.try_set_result(None))
    return f


class ReadySet:
    """Container for an ordered set of tasks that are ready to advance."""

    def __init__(self):
        self._cond = threading.Condition()
        self._buffer = futuretools.BufferGroup()
        self._tasks: List[Task] = []
        self._task_set: Set[Task] = set()

    def register(self, task: Task) -> None:
        """Registers task to be added to this set when it is ready."""
        self._buffer.add(task.future)
        task.add_ready_callback(self._add)

    def _add(self, task: Task) -> None:
        """Adds the given task to the ready set, if it is not already there."""
        with self._cond:
            if task not in self._task_set:
                self._task_set.add(task)
                self._tasks.append(task)
                self._cond.notify()

    def get_all(self, timeout: Optional[float] = None) -> List[Task]:
        """Gets all ready tasks and clears the ready set.

        If no tasks are ready yet, we flush buffered futures to notify them
        that they should proceed, and then block until one or more tasks become
        ready.

        Raises:
            ValueError if timeout is < 0 or > threading.TIMEOUT_MAX
        """
        if timeout is not None and (timeout < 0 or timeout > threading.TIMEOUT_MAX):
            raise ValueError(f"invalid timeout: {timeout}")
        with self._cond:
            if self._tasks:
                return self._pop_tasks()
        # Flush buffered futures to ensure we make progress. Note that we must
        # release the condition lock before flushing to avoid a deadlock if
        # buffered futures complete and trigger a call to self._add.
        self._buffer.flush()
        with self._cond:
            if not self._tasks:
                if not self._cond.wait(timeout):
                    raise TimeoutError()
            return self._pop_tasks()

    def _pop_tasks(self) -> List[Task]:
        tasks = self._tasks
        self._tasks = []
        self._task_set.clear()
        return tasks

    def interrupt(self) -> None:
        with self._cond:
            self._cond.notify()


@functools.total_ordering
class DeadlineEntry:
    """A entry for one Deadline in the Scheduler's priority queue.

    This follows the implementation notes in the stdlib heapq docs:
    https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes

    Attributes:
        task: The task associated with this deadline.
        deadline: Absolute time when the deadline will elapse.
        count: Monotonically-increasing counter to preserve creation order when
            comparing entries with the same deadline.
        valid: Flag indicating whether the deadline is still valid. If the task
            exits its scope before the deadline elapses, we mark the deadline as
            invalid but leave it in the scheduler's priority queue since removal
            would require an O(n) scan. The scheduler ignores invalid deadlines
            when they elapse.
    """

    _counter = itertools.count()

    def __init__(self, task: Task, deadline: float, timeout_error: TimeoutError):
        self.task = task
        self.deadline = deadline
        self.timeout_error = timeout_error
        self.count = next(self._counter)
        self._cmp_val = (deadline, self.count)
        self.valid = True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeadlineEntry):
            return NotImplemented
        return self._cmp_val == other._cmp_val

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeadlineEntry):
            return NotImplemented
        return self._cmp_val < other._cmp_val

    def __repr__(self) -> str:
        return f"DeadlineEntry({self.task}, {self.deadline}, {self.count})"


class Scheduler:
    def __init__(self) -> None:
        self.active_tasks: Set[Task] = set()
        self._ready_tasks = ReadySet()
        self._prev_signal: Optional[Callable] = None
        self._interrupted = False
        self._deadlines: List[DeadlineEntry] = []

    def spawn(self, awaitable: Awaitable[Any], main_task: Optional[Task] = None) -> Task:
        """Spawns a new Task to run an awaitable in this Scheduler.

        Note that the task will not be advanced until the next scheduler tick.
        Also, note that this function is safe to call from sync code (such as
        duet.run) or async code (such as within a scope).

        Args:
            func: The async function to run.
            *args: Args for func.
            **kwds: Keyword args for func.

        Returns:
            A Task to run the given awaitable.
        """
        task = Task(awaitable, scheduler=self, main_task=main_task)
        self.active_tasks.add(task)
        self._ready_tasks.register(task)
        return task

    def time(self) -> float:
        return time.time()

    def add_deadline(self, entry: DeadlineEntry) -> None:
        heapq.heappush(self._deadlines, entry)

    def get_next_deadline(self) -> Optional[float]:
        while self._deadlines:
            if not self._deadlines[0].valid:
                heapq.heappop(self._deadlines)
                continue
            return self._deadlines[0].deadline
        return None

    def get_deadline_entries(self, deadline: float) -> Iterator[DeadlineEntry]:
        while self._deadlines and self._deadlines[0].deadline <= deadline:
            entry = heapq.heappop(self._deadlines)
            if entry.valid:
                yield entry

    def tick(self):
        """Runs the scheduler ahead by one tick.

        This waits for at least one active task to complete, then advances all
        ready tasks and sets up a new future to be notified later by tasks that
        are still active (or yet to be spawned). Raises a RuntimeError if there
        are no currently active tasks.
        """
        if not self.active_tasks:
            raise RuntimeError("tick called with no active tasks")

        if self._interrupted:
            task = next(iter(self.active_tasks))
            task.interrupt(task, KeyboardInterrupt)
            self._interrupted = False

        deadline = self.get_next_deadline()
        if deadline is None:
            ready_tasks = self._ready_tasks.get_all(None)
        else:
            ready_tasks: List[Task] = []
            for i in itertools.count():
                timeout = deadline - self.time()
                if i and timeout < 0:
                    break
                try:
                    ready_tasks = self._ready_tasks.get_all(
                        min(0, max(timeout, threading.TIMEOUT_MAX))
                    )
                    break
                except TimeoutError:
                    pass
            if not ready_tasks:
                for entry in self.get_deadline_entries(deadline):
                    entry.task.interrupt(entry.task, entry.timeout_error)
                ready_tasks = self._ready_tasks.get_all(None)
        for task in ready_tasks:
            try:
                task.advance()
            finally:
                if task.done:
                    task.close()
                    self.active_tasks.discard(task)
                else:
                    self._ready_tasks.register(task)

    def _interrupt(self, signum: int, frame: Optional[Any]) -> None:
        """Interrupt signal handler used while this scheduler is running.

        This is inspired by trio's interrupt handling, described here:
        https://vorpus.org/blog/control-c-handling-in-python-and-trio/

        If the interrupted frame is inside a running task, which we detect by
        looking for a special local variable inserted into the task coroutine,
        we simply raise a KeyboardInterrupt as usual. Otherwise we set a flag
        which will get checked on the next tick() and cause a task to be
        interrupted.

        One important difference from trio is that duet is reentrant, so when
        detecting whether we are in a task we have to check whether the task's
        scheduler is self. If the interrupted frame is running in a task of a
        different scheduler, that should not raise KeyboardInterrupt directly.
        """
        if self._in_task(frame):
            raise KeyboardInterrupt
        else:
            self._interrupted = True
            self._ready_tasks.interrupt()

    def _in_task(self, frame) -> bool:
        while frame is not None:
            if frame.f_locals.get(LOCALS_TASK_SCHEDULER, None) is self:
                return True
            frame = frame.f_back
        return False

    def __enter__(self):
        if (
            threading.current_thread() == threading.main_thread()
            and signal.getsignal(signal.SIGINT) == signal.default_int_handler
        ):
            self._prev_signal = signal.signal(signal.SIGINT, self._interrupt)
        return self

    def __exit__(self, exc_type, exc, tb):
        def finish_tasks(error=None):
            if error:
                for task in self.active_tasks:
                    task.interrupt(None, error)
            while self.active_tasks:
                try:
                    self.tick()
                except Exception:
                    if not error:
                        raise

        try:
            if exc:
                finish_tasks(exc)
            else:
                try:
                    finish_tasks()
                except Exception as exc:
                    finish_tasks(exc)
                    raise
        finally:
            if self._prev_signal:
                signal.signal(signal.SIGINT, self._prev_signal)
