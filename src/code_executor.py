# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions

import multiprocessing
import queue
import subprocess
import sys
import time
import traceback

import constants

multiprocessing.set_start_method("fork", force=True)
# WARNING
# This program exists to execute untrusted model-generated code. Although
# it is highly unlikely that model-generated code will do something overtly
# malicious in response to this test suite, model-generated code may act
# destructively due to a lack of model capability or alignment.
# Users are strongly encouraged to sandbox this evaluation suite so that it
# does not perform destructive actions on their host or network.
# Proceed at your own risk:

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class CodeExecutor:
    """A class to execute code in a separate process."""

    def __init__(self):
        self._reset_execution_time()

    def _add_execution_time(self, time_taken: float):
        """
        Add the time taken to the running average.

        Args:
            time_taken (float): The time taken to execute the program.
        """
        self.last_program_execution_count += 1
        self.last_program_execution_time = time_taken
        self.last_program_execution_time_mean = (
            self.last_program_execution_time_mean * self.last_program_execution_count
            + time_taken
        ) / (self.last_program_execution_count + 1)

    def _reset_execution_time(self):
        """
        Reset the execution time statistics.
        """
        self.last_program_execution_time = 0.0
        self.last_program_execution_count = 0
        self.last_program_execution_time_mean = 0.0

    def _exec_program(self, q, program, input_data, expected_output, timeout):
        """
        Execute the program in a separate Python process.

        Args:
            q (multiprocessing.Queue): The queue to store the result of the execution.
            program (str): The program to execute.
            input_data (str): The input data to provide to the program.
            expected_output (str): The expected output of the program.
            timeout (float): The maximum time to allow for execution.
        """
        try:
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, "-c", program],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=constants.CHAR_ENCODING_UTF8,
            )
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            if time.time() - start_time > timeout:
                raise TimeoutError(constants.EXECUTOR_MESSAGE__TIMEDOUT)
            if process.returncode != 0:
                q.put(f"{constants.EXECUTOR_MESSAGE__FAILED}: {stderr}")
            else:
                if stdout.strip() == expected_output.strip():
                    q.put(constants.EXECUTOR_MESSAGE__PASSED)
                else:
                    q.put(
                        f"{constants.EXECUTOR_MESSAGE__WRONG_ANSWER}. Expected '{expected_output}', got '{stdout}'"
                    )
        except subprocess.TimeoutExpired:
            process.kill()
            q.put(constants.EXECUTOR_MESSAGE__TIMEDOUT)
        except Exception:
            q.put(f"{constants.EXECUTOR_MESSAGE__FAILED}: {traceback.format_exc()}")
        finally:
            self._add_execution_time(time.time() - start_time)

    def check_correctness(
        self,
        program: str,
        input_data: str,
        expected_output: str,
        timeout: float,
        reset_execution_time: bool = False,
    ) -> str:
        """
        Check the correctness of the program by executing it as a separate Python process.

        Args:
            program (str): The program to execute.
            input_data (str): The input data to provide to the program.
            expected_output (str): The expected output of the program.
            timeout (float): The maximum time to allow for execution.
            reset_execution_time (bool): Whether to reset the execution time statistics.

        Returns:
            str: The result of the execution.
        """
        if reset_execution_time:
            self._reset_execution_time()
        q = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._exec_program,
            args=(q, program, input_data, expected_output, timeout),
        )
        process.start()
        process.join(timeout=timeout + 1)
        if process.is_alive():
            process.terminate()
            process.join()
            result = constants.EXECUTOR_MESSAGE__TIMEDOUT
        else:
            try:
                result = q.get_nowait()
            except queue.Empty:
                result = constants.EXECUTOR_MESSAGE__NO_RESULTS
            finally:
                ic(input_data, expected_output, result)
        return result
