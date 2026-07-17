"""
Simple file + console logger for the batch pipeline.
"""
import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("fraud_pipeline")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("logs/pipeline.log")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)