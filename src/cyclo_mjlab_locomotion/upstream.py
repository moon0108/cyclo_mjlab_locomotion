# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load cyclo_mjlab's reinforcement learning scripts, which are not installed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import source


def load_upstream_script(name: str) -> ModuleType:
  """Import scripts/reinforcement_learning/<name>.py next to `source`."""
  root = Path(source.__file__).resolve().parents[1]
  path = root / "scripts" / "reinforcement_learning" / f"{name}.py"
  module_name = f"cyclo_mjlab_{name}"
  spec = importlib.util.spec_from_file_location(module_name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module  # Dataclasses resolve types through this.
  spec.loader.exec_module(module)
  return module
