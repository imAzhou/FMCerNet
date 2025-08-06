from .classifier.binary_linear import BinaryLinear
from .classifier.multicls_linear import MultiClsLinear
from .classifier.mlc_linear import MLCLinear
from .classifier.chief import CHIEF
from .classifier.ml_decoder import MLDecoder
from .classifier.query2label import Query2Label
from .classifier.SVT_classifier import SVTClassifier
from .classifier.wscer_mlc import WSCerMLC
from .classifier.wscer_binary import WSCerBinary
from .classifier.wscer_partial import WSCerPartial
from .classifier.wscer_alltoken import WSCerAllToken

allowed_classifier_type = ['online_version','binary_linear', 'multicls_linear', 'mlc_linear', 'chief', 'ml_decoder', 'query2label', 'svt',
                           'wscer_mlc', 'wscer_binary', 'wscer_partial', 'wscer_alltoken']

def get_classifier(args):
    classifier_type = args.taskhead_model
    assert classifier_type in allowed_classifier_type, f'classifier_type allowed in {allowed_classifier_type}'
    
    classifier = None
    if classifier_type == 'binary_linear':
        classifier = BinaryLinear
    if classifier_type == 'multicls_linear':
        classifier = MultiClsLinear
    if classifier_type == 'mlc_linear':
        classifier = MLCLinear
    if classifier_type == 'chief':
        classifier = CHIEF
    if classifier_type == 'ml_decoder':
        classifier = MLDecoder
    if classifier_type == 'query2label':
        classifier = Query2Label
    if classifier_type == 'svt':
        classifier = SVTClassifier

    if classifier_type == 'wscer_mlc':
        classifier = WSCerMLC
    if classifier_type == 'wscer_binary':
        classifier = WSCerBinary
    if classifier_type == 'wscer_partial':
        classifier = WSCerPartial
    if classifier_type == 'wscer_alltoken':
        classifier = WSCerAllToken
    
    return classifier(args)