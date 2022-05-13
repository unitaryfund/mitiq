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

import duet
import duet.impl as impl


class CompleteOnFlush(duet.BufferedFuture):
    def __init__(self):
        super().__init__()
        self.flushed = False

    def flush(self):
        self.flushed = True
        self.set_result(None)


def make_task(future: duet.AwaitableFuture) -> impl.Task:
    """Make a task from the given future.

    We advance the task once, which just starts the generator and yields the
    future itself.
    """
    task = impl.Task(future, impl.Scheduler(), None)
    task.advance()
    return task


class TestReadySet:
    def test_get_all_returns_all_ready_tasks(self):
        task1 = make_task(duet.completed_future(None))
        task2 = make_task(duet.completed_future(None))
        task3 = make_task(duet.AwaitableFuture())
        task4 = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task1)
        rs.register(task2)
        rs.register(task3)
        rs.register(task4)
        tasks = rs.get_all()
        assert tasks == [task1, task2, task4]

    def test_task_added_at_most_once(self):
        task = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task)
        rs.register(task)
        tasks = rs.get_all()
        assert tasks == [task]

    def test_futures_flushed_if_no_task_ready(self):
        future = CompleteOnFlush()
        task = make_task(future)
        rs = impl.ReadySet()
        rs.register(task)
        tasks = rs.get_all()
        assert tasks == [task]
        assert future.flushed

    def test_futures_not_flushed_if_tasks_ready(self):
        future = CompleteOnFlush()
        task1 = make_task(future)
        task2 = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task1)
        rs.register(task2)
        tasks = rs.get_all()
        assert tasks == [task2]
        assert not future.flushed
