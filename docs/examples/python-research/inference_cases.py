"""Composition, mapping and refusal cases using the evaluated classifier Sensors."""
from dataclasses import asdict
import pickle

import numpy as np
from limen import Cohort


# region composition
def compose_selected(sensors, bars):
    def record(row):
        value = asdict(row)
        value['datetime'] = row.datetime.isoformat() if row.datetime is not None else None
        return value

    history = bars[1428:1920]
    before = [pickle.dumps(sensor.fitted_state) for sensor in sensors]
    members = [sensor.predict_all(history) for sensor in sensors]
    mean = Cohort.from_python(sensors, aggregate='mean', state_id='calibrated-mean')
    vote = Cohort.from_python(sensors, aggregate='vote', state_id='calibrated-vote')
    signed = Cohort.from_python(sensors, prepare=lambda context, member: context['bars'],
        map_output=lambda member, row: (2 * row.prediction - 1, None), mapped_output_id='next_bar_signed_direction',
        aggregate=lambda rows: (float(np.median([row.prediction for row in rows])), None), state_id='signed-median')
    later = Cohort.from_python(sensors, selection_cutoff=bars['datetime'][1441], state_id='later-selection')
    result = {'members': [[record(row) for row in tape] for tape in members],
              'mean': [record(row) for row in mean.predict_all(history)],
              'vote': [record(row) for row in vote(history)],
              'signed': [record(row) for row in signed.predict_all({'bars': history})],
              'later': [record(row) for row in later.predict_all(history)],
              'last': record(mean.predict(history)), 'empty': mean.predict_all(history.head(0))}
    if before != [pickle.dumps(sensor.fitted_state) for sensor in sensors]:
        raise ValueError('Inference changed the evaluated fitted state')
    refusals = {}
    for name, call in {'duplicate': lambda: Cohort.from_python([sensors[0], sensors[0]]),
                       'missing_mapping_identity': lambda: Cohort.from_python(sensors, map_output=lambda member, row: (row.prediction, row.probability)),
                       'unaligned_members': lambda: Cohort.from_python(sensors, prepare=lambda data, member: data if member is sensors[0] else data.tail(data.height-1)).predict_all(history),
                       'reverse_decisions': lambda: mean.predict_all(history.reverse())}.items():
        try:
            call()
        except ValueError as error:
            refusals[name] = str(error)
        else:
            raise ValueError('An invalid composition was accepted: ' + name)
    # Results retain real UTC timestamps and declared availability, serialized by the caller.
    return {'results': result, 'refusals': refusals, 'retained_state_unchanged': True}
# endregion composition
