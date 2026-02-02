import logging
import json

#setting up logger
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def safe_get_confidence(entry):
    try:
        confidence = float(entry.get('confidence'))
        return confidence
    except (ValueError, TypeError):
        logger.warning(f'Confidence value {entry.get("confidence")} cant be converted to float')
        return 0

def data_aggregator(validated_data):
    aggregated_data = {}
    for entry in validated_data:
        imgID = entry.get('imgID')
        if imgID not in aggregated_data:
            aggregated_data[imgID] = {}
            aggregated_data[imgID]['detections'] = [entry.get('label')]
            aggregated_data[imgID]['max_conf'] = safe_get_confidence(entry)
            aggregated_data[imgID]['count'] = 1

        else:
            if entry.get('label') not in aggregated_data[imgID]['detections']:
                aggregated_data[imgID]['detections'].append(entry.get('label'))
            if aggregated_data[imgID]['max_conf'] < safe_get_confidence(entry):
                aggregated_data[imgID]['max_conf'] = safe_get_confidence(entry)
            aggregated_data[imgID]['count'] += 1

    with open('aggregated_data.json', 'w') as f:
        json.dump(aggregated_data, f, indent=4)





