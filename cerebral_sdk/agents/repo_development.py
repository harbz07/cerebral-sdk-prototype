"""Repository development agent primitives."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class DevelopmentTask:
    """Single development task for an implementation cycle."""

    task_id: str
    title: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Result of implementing and validating a single task."""

    task_id: str
    status: str
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepositoryDevelopmentAgent:
    """Runs implementation and validation loops over repository tasks."""

    def __init__(
        self,
        implementer: Callable[[DevelopmentTask], TaskResult],
        validator: Optional[Callable[[TaskResult], TaskResult]] = None,
    ):
        self.implementer = implementer
        self.validator = validator

    def run_once(self, tasks: Sequence[DevelopmentTask]) -> List[TaskResult]:
        """Run a single implementation cycle."""
        results: List[TaskResult] = []
        for task in tasks:
            result = self.implementer(task)
            if self.validator:
                result = self.validator(result)
            results.append(result)
        return results

    def run_continuously(
        self,
        task_source: Callable[[], Iterable[DevelopmentTask]],
        max_cycles: Optional[int] = None,
    ) -> List[TaskResult]:
        """Run implementation cycles continuously from a task source."""
        cycle = 0
        all_results: List[TaskResult] = []

        while max_cycles is None or cycle < max_cycles:
            tasks = list(task_source())
            if not tasks:
                break

            all_results.extend(self.run_once(tasks))
            cycle += 1

        return all_results
