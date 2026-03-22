from .model import ModelBase, load_model
from .intervention import get_intervention_func
from .steering_utils import *


def extract_candidate_vectors(*args, **kwargs):
    from .extract import extract_candidate_vectors as _extract_candidate_vectors
    return _extract_candidate_vectors(*args, **kwargs)


def validate(*args, **kwargs):
    from .validate import validate as _validate
    return _validate(*args, **kwargs)
