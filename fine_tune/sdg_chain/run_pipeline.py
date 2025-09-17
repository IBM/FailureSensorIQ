import json
from pipeline import SensorPG, FailureModePG, Sensor2FMPG, FM2SensorPG
from tqdm.contrib.concurrent import process_map  # or thread_map
import os


def generate_sensor_problems():
    with open('../../data/asset_kb_candicate.json') as f:
        kb_candicate = json.load(f)

    sensor_candidates_mapping = kb_candicate['sensor_candidates']
    for group in sensor_candidates_mapping:
        print("group", group)
        mapping = sensor_candidates_mapping[group]
        assets = mapping['assets']
        sensor_candidates = mapping['sensor_candidates']
        inputs = [{'sensor_candidates': sensor_candidates, 'asset_class': asset} for asset in assets]
        r = process_map(SensorPG().generate, inputs, max_workers=5)


def generate_failure_mode_problems():
    with open('../../data/asset_kb_candicate.json') as f:
        kb_candicate = json.load(f)

    failure_mode_candidates_mapping = kb_candicate['failure_mode_candidates']
    for group in failure_mode_candidates_mapping:
        print("group", group)
        mapping = failure_mode_candidates_mapping[group]
        assets = mapping['assets']
        failure_mode_candidates = mapping['failure_mode_candidates']
        inputs = [{'failure_mode_candidates': failure_mode_candidates, 'asset_class': asset} for asset in assets]
        r = process_map(FailureModePG().generate, inputs, max_workers=5)


def generate_sensor2fm_problems():
    with open('../../data/asset_kb_candicate.json') as f:
        kb_candicate = json.load(f)
    with open('../../data/asset_kb_fact.json') as f:
        kb_fact = json.load(f)

    relevant_sensors_for_each_asset = kb_fact['relevant_sensors_for_each_asset']
    failure_mode_candidates_mapping = kb_candicate['failure_mode_candidates']
    for group in failure_mode_candidates_mapping:
        print("group", group)
        mapping = failure_mode_candidates_mapping[group]
        assets = mapping['assets']
        failure_mode_candidates = mapping['failure_mode_candidates']
        # skip generated assets
        # assets = [ asset for asset in assets if not os.path.exists(f'./generations/sensor2fm/{asset}.json')]
        
        inputs = [{'failure_mode_candidates': failure_mode_candidates,
                   'relevant_sensors_for_each_asset': relevant_sensors_for_each_asset,
                   'asset_class': asset} for asset in assets]
        r = process_map(Sensor2FMPG().generate, inputs, max_workers=8)


def generate_fm2sensor_problems():
    with open('../../data/asset_kb_candicate.json') as f:
        kb_candicate = json.load(f)
    with open('../../data/asset_kb_fact.json') as f:
        kb_fact = json.load(f)

    relevant_fm_for_each_asset = kb_fact['relevant_fm_for_each_asset']
    sensor_candidates_mapping = kb_candicate['sensor_candidates']
    for group in sensor_candidates_mapping:
        print("group", group)
        mapping = sensor_candidates_mapping[group]
        assets = mapping['assets']
        sensor_candidates = mapping['sensor_candidates']
        # skip generated assets
        # assets = [ asset for asset in assets if not os.path.exists(f'./generations/fm2sensor/{asset}.json')]
        
        inputs = [{'sensor_candidates': sensor_candidates,
                   'relevant_failure_modes_for_each_asset': relevant_fm_for_each_asset, 'asset_class': asset}
                  for asset in assets]
        r = process_map(FM2SensorPG().generate, inputs, max_workers=8)


if __name__ == '__main__':
    # generate_sensor_problems()
    # generate_failure_mode_problems()
    # generate_sensor2fm_problems()
    generate_fm2sensor_problems()
