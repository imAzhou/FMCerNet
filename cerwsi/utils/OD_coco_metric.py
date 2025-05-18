from collections import defaultdict
import json
from sklearn.metrics import accuracy_score, confusion_matrix
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
import numpy as np
import re
from mmengine.logging import MMLogger
from mmdet.evaluation import CocoMetric