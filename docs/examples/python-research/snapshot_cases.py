"""Capture explicit caller-owned fitted state separately from checkpoint recovery."""
import hashlib

import numpy as np
from limen import UniversalExperimentLoop

import research as project


# region captured_state
def captured_state(bars, output):
    output.mkdir(parents=True)
    loop, retained = project.run_trials(bars[:1440], str(output / 'captured-rounds'))
    loop.extras.append({'fitted_models': retained})
    path = loop.save(output / 'captured.uel')
    restored = UniversalExperimentLoop.load(path)
    assert isinstance(restored, UniversalExperimentLoop)
    if not project.completed_log(restored).equals(project.completed_log(loop)):
        raise ValueError('Captured result log changed during restoration')
    validation = project.prepare_split(bars[960:1440], {'window': 4})
    forecasts = []
    for (window, alpha), fitted in restored.extras[-1]['fitted_models'].items():
        validation = project.prepare_split(bars[960:1440], {'window': window})
        x = validation.select('momentum')
        values = fitted.predict(x)
        if not np.array_equal(values, retained[(window, alpha)].predict(x)):
            raise ValueError('Captured fitted predictions changed during restoration')
        forecasts.append({'window': window, 'alpha': alpha, 'predictions': values.tolist()})
    return {'rounds': project.completed_log(restored).height, 'models': len(forecasts),
            'forecast_rows': validation.height, 'forecasts': forecasts,
            'snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'restored_fitted_state': True}
# endregion captured_state
