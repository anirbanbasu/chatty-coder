# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow executor engine."""

from typing import List
import dirtyjson as json
import sys
from tqdm import tqdm

from llama_index.core.llms.llm import LLM

from llama_index.core.workflow import (
    Workflow,
)

from llama_index.core.workflow.handler import WorkflowHandler

from constants import (
    EMPTY_STRING,
    PROJECT_NAME,
    PYDANTIC_MODEL__CODE_OUTPUT__CODE,
    PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE,
    PYDANTIC_MODEL__CODE_OUTPUT__REASONING,
)
from utils import get_terminal_size
from workflows.common import (
    TestCase,
    WorkflowStatusEvent,
)
from workflows.self_reflective_coder import CompetitiveCoderWorkflow

from llama_index.core.workflow.events import (
    HumanResponseEvent,
)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class ChattyCoderEngine:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm
        self.available_workflows = [
            CompetitiveCoderWorkflow,
        ]

    def get_workflows_dataframe(self) -> list[list[str, str]]:
        """
        Get the dataframe of available workflows.

        Returns:
            list[list[str, str]]: The list of available workflows with their names and descriptions.
        """
        return [
            [f"_{workflow.__name__}_", workflow.__doc__]
            for workflow in self.available_workflows
        ]

    def get_workflow_description(self, workflow_name: str) -> str:
        """
        Get the description of the workflow with the given name.

        Args:
            workflow_name (str): The name of the workflow to get the description for.

        Returns:
            str: The description of the workflow.

        Raises:
            ValueError: If the workflow with the given name is not supported.
        """
        for workflow in self.available_workflows:
            if workflow.__name__ == workflow_name:
                return workflow.__doc__
        raise ValueError(f"Workflow with name '{workflow_name}' is not supported.")

    def get_workflow_names_list(self) -> list[str]:
        """
        Get the list of available workflow names.

        Returns:
            list[str]: The list of available workflow names.
        """
        return [workflow.__name__ for workflow in self.available_workflows]

    def get_workflow_by_name(self, workflow_name: str):
        """
        Get the workflow with the given name.

        Args:
            workflow_name (str): The name of the workflow to get.

        Returns:
            Any: The workflow with the given name.

        Raises:
            ValueError: If the workflow with the given name is not supported.
        """
        for workflow in self.available_workflows:
            if workflow.__name__ == workflow_name:
                return workflow
        raise ValueError(f"Workflow with name '{workflow_name}' is not supported.")

    def send_human_response(self, response: str):
        if (
            self._running_workflow_handler is not None
            and not self._running_workflow_handler.is_done()
        ):
            # self._running_workflow_handler.ctx.send_event(
            #     HumanResponseEvent(response=response)
            # )
            self._human_response = response
            # self._human_input_available_indicator.state = (
            #     HumanInputAvailableIndicatorTask.STATE_DONE
            # )
            self._running_workflow_handler.ctx.send_event(
                HumanResponseEvent(response=self._human_response)
            )

    async def run(
        self,
        problem: str,
        test_cases: List[TestCase] = None,
        runtime_limit: float = 30.0,
        workflow_name: str = CompetitiveCoderWorkflow.__name__,
    ):
        chosen_workflow = self.get_workflow_by_name(workflow_name)

        # Should the timeout be None for human-in-the-loop workflows?
        workflow_init_kwargs = {"llm": self.llm, "timeout": 180, "verbose": True}
        workflow_run_kwargs = {}

        if chosen_workflow in [
            CompetitiveCoderWorkflow,
        ]:
            workflow_run_kwargs["problem"] = problem
            workflow_run_kwargs["test_cases"] = test_cases
            workflow_run_kwargs["runtime_limit"] = runtime_limit
        else:
            raise ValueError(f"Workflow '{workflow_name}' is not supported.")

        workflow: Workflow = chosen_workflow(**workflow_init_kwargs)
        print(
            f"\nAttempting the question using the {workflow_name} workflow. This may take a while...",
            flush=True,
        )

        self._running_workflow_handler: WorkflowHandler = workflow.run(
            **workflow_run_kwargs
        )
        done: bool = False
        total_steps: int = 0
        finished_steps: int = 0
        terminal_columns, _ = get_terminal_size()
        progress_bar = tqdm(
            total=total_steps,
            leave=False,
            unit="step",
            ncols=int(terminal_columns / 2),
            desc=PROJECT_NAME,
            colour="yellow",
        )
        async for event in self._running_workflow_handler.stream_events():
            if isinstance(event, WorkflowStatusEvent):
                total_steps = event.total_steps
                finished_steps = event.finished_steps
                print(f"\n{str(event.msg)}", flush=True)
                progress_bar.reset(total=total_steps)
                progress_bar.update(finished_steps)
                progress_bar.refresh()
            yield done, finished_steps, total_steps, event
        try:
            # done_tasks, pending_tasks = await asyncio.wait(
            #     [self._running_workflow_handler],
            #     timeout=runtime_limit,
            # )
            # done = self._running_workflow_handler.is_done()
            # if done_tasks and done:
            #     result = json.loads(self._running_workflow_handler.result())

            # for task in pending_tasks:
            #     ic(task)
            result = json.loads(await self._running_workflow_handler)
            done = True
        except Exception as e:
            result = {
                PYDANTIC_MODEL__CODE_OUTPUT__REASONING: f"\nException in running the workflow(s). Type: {type(e).__name__}. Message: '{str(e)}'",
                PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE: EMPTY_STRING,
                PYDANTIC_MODEL__CODE_OUTPUT__CODE: EMPTY_STRING,
            }
            # Set this to done, otherwise another workflow call cannot be made.
            done = True
            print(result, file=sys.stderr, flush=True)
            raise e
        finally:
            progress_bar.close()
            yield done, finished_steps, total_steps, result
