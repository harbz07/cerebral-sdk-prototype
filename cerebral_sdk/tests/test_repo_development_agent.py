"""Tests for continuous repository development agents."""

from cerebral_sdk.agents import (
    DevelopmentTask,
    RepositoryDevelopmentAgent,
    TaskResult,
)


def test_run_once_implements_each_task():
    tasks = [
        DevelopmentTask(task_id="1", title="Implement API"),
        DevelopmentTask(task_id="2", title="Add docs"),
    ]

    def implementer(task: DevelopmentTask) -> TaskResult:
        return TaskResult(task_id=task.task_id, status="implemented", summary=task.title)

    agent = RepositoryDevelopmentAgent(implementer=implementer)
    results = agent.run_once(tasks)

    assert [r.task_id for r in results] == ["1", "2"]
    assert all(r.status == "implemented" for r in results)


def test_run_continuously_applies_validator_across_cycles():
    queue = [
        [DevelopmentTask(task_id="1", title="Task 1")],
        [DevelopmentTask(task_id="2", title="Task 2")],
    ]

    def source():
        return queue.pop(0) if queue else []

    def implementer(task: DevelopmentTask) -> TaskResult:
        return TaskResult(task_id=task.task_id, status="implemented", summary=task.title)

    def validator(result: TaskResult) -> TaskResult:
        return TaskResult(
            task_id=result.task_id,
            status="validated",
            summary=result.summary,
            metadata={"validated": True},
        )

    agent = RepositoryDevelopmentAgent(implementer=implementer, validator=validator)
    results = agent.run_continuously(task_source=source, max_cycles=5)

    assert [r.task_id for r in results] == ["1", "2"]
    assert all(r.status == "validated" for r in results)
