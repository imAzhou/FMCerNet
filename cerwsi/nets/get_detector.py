from .detector.detr import WSDETR

allowed_classifier_type = ['detr',]

def get_detector(args):
    detector_type = args.taskhead_model
    assert detector_type in allowed_classifier_type, f'detector_type allowed in {allowed_classifier_type}'
    
    detector = None
    if detector_type == 'detr':
        detector = WSDETR
    
    return detector(args)