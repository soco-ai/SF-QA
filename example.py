import argparse
import json
import logging

import soco_openqa.helper as helper
from soco_openqa.pipeline import OpenQA
from soco_openqa.config import get_config
from soco_openqa.evaluation import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help='yaml file name')
    args = parser.parse_args()

    config = get_config(args.config)
    data = helper.load_json(file_dir=config.data.name, file_name=config.data.split)

    # Initiate evaluation pipeline
    qa = OpenQA(config)
    predictions = qa.predict(data)

    # Evaluate predictions
    results = evaluate(config.data.lang, data, predictions, config)
    logger.info(results)

    # Save locally
    helper.save_logs(config.dump(), results, save_name=config.config_name)
