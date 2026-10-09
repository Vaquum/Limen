import json
from collections.abc import Mapping

from limen.experiment.manifest_core import Manifest


def _result_params(params: Mapping[str, object], prepared: Mapping[str, object], manifest: Manifest | None) -> dict[str, object]:
    result = dict(params)
    if getattr(manifest, 'ablation_config', None) is not None:
        result['_dropped_features'] = json.dumps(prepared.get('_dropped_features', []))
    return result
