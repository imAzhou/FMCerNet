# from .MIL.HMIL import HMIL
# from .MIL.TransMIL import TransMIL

allowed_mil_type = ['HMIL', 'TransMIL']

def get_mil(args):
    mil_type = args.mil_type
    assert mil_type in allowed_mil_type, f'mil_type allowed in {allowed_mil_type}'
    
    mil_model = None
    # if mil_type == 'HMIL':
    #     mil_model = HMIL
    # if mil_type == 'TransMIL':
    #     mil_model = TransMIL
    
    return mil_model(args)