import csv
import os
import gzip
import pickle as pkl
import json

from abc import abstractmethod
import numpy as np
import hashlib
from tqdm import tqdm
import torch
import logging

logger = logging.getLogger(__name__)

def clean_web_text(s):
    return s.replace('&apos;', '\'').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<'). \
        replace('&gt;', '>').replace('&#91;', '[').replace('&#93;', ']')

def align_column(row):
    row_str = ''
    for i, item in enumerate(row):
        if 'float' in item.__class__.__name__:
            item = f'{item:.4f}'
        if i == 0:
            row_str += f'{item:>12}'
        else:
            row_str += f'{item:>10}'
    return row_str


def report_results(header, results, axis):
    n_column = len(header)
    metric = header[axis].split('_')[-1]
    if metric in {'acc', 'f1', 'mcc', 'pearson'}:
        cmp = lambda x1, x2: x1 < x2
        best_row = [0] * n_column
    elif metric in {'loss', 'ppl'}:
        cmp = lambda x1, x2: x1 > x2
        best_row = [10000] * n_column
    else:
        raise NotImplementedError
    logger.info('')
    logger.info(align_column(header))
    if results[0][0] == 'before':
        before_row = results[0]
        results = results[1:]
        logger.info(align_column(before_row))
    else:
        before_row = None
    logger.info('-' * (n_column * 10 + 2))
    for row in results:
        logger.info(align_column(row))
        if cmp(best_row[axis], row[axis]):
            best_row = row
    logger.info('-' * (n_column * 10 + 2))
    if metric in {'acc', 'f1', 'mcc', 'pearson'}:
        overfit = results[-1][axis] < best_row[axis] - 0.01
    elif metric in {'loss', 'ppl'}:
        overfit = best_row[axis] + 0.01 < results[-1][axis]
    else:
        raise NotImplementedError
    logger.info(align_column([f'best: {best_row[0]}'] + best_row[1:] + (['(overfit)'] if overfit else [])))
    if before_row is not None:
        logger.info(align_column(['gain'] + [best - before for (best, before) in zip(best_row[1:], before_row[1:])]))
    return best_row

def batch_iter(dataloader, n_epochs=-1):
    if n_epochs == -1:
        while True:
            for batch in dataloader:
                yield batch
    else:
        for _ in range(n_epochs):
            for batch in dataloader:
                yield batch

if __name__ == '__main__':
    tokenize('sst2')
    print()
    tokenize('sst5')
    print()
    tokenize('imdb')
    print()
    tokenize('yelp2')
    print()
    tokenize('yelp5')
