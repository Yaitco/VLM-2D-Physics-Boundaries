from __future__ import annotations

import os
from typing import Any, Dict, Optional


def get_secret_value(secret_name: str) -> Optional[str]:
    try:
        from google.colab import userdata

        value = userdata.get(secret_name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(secret_name.upper()) or os.getenv(secret_name)


def init_comet_experiment(
    run_tag: str,
    run_params: Dict[str, Any],
    enabled: bool = True,
    default_project: str = 'vlm-physics-validation',
):
    if not enabled:
        print('Comet: disabled.')
        return None

    api_key = get_secret_value('comet_api_key')
    workspace = get_secret_value('comet_workspace')
    project_name = get_secret_value('comet_project_name') or default_project

    if not api_key:
        print("Comet: API key is missing. Add 'comet_api_key' in Colab userdata or set COMET_API_KEY.")
        return None

    try:
        from comet_ml import Experiment
    except Exception as exc:
        print(f'Comet: comet_ml import failed: {exc}')
        return None

    kwargs = {
        'api_key': api_key,
        'project_name': project_name,
        'auto_output_logging': 'simple',
    }
    if workspace:
        kwargs['workspace'] = workspace

    try:
        exp = Experiment(**kwargs)
    except Exception as exc:
        print(f'Comet: failed to initialize Experiment: {exc}')
        return None

    exp.set_name(run_tag)
    exp.log_parameters(run_params)
    print(f"Comet: logging to project='{project_name}' workspace='{workspace}'")
    return exp
