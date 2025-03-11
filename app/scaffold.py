"""
Adapted from Bardes, A. et al. Revisiting feature prediction for learning visual
representations from video. arXiv [cs.CV] (2024).; 
https://github.com/facebookresearch/jepa/blob/main/app/scaffold.py (10.03.2025).
"""


import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):

    logger.info(f'Running pre-training of app: {app}')
    return importlib.import_module(f'app.{app}.train').main(
        args=args,
        resume_preempt=resume_preempt)