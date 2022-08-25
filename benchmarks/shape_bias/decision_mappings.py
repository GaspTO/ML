import numpy as np
from benchmarks.shape_bias.human_categories import get_human_object_recognition_categories, HumanCategories
from benchmarks.shape_bias.wordnet_functions import get_ilsvrc2012_WNIDs, get_ilsvrc2012_categories


def check_input(probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

def imagnet_probabilities_to_1000class_wnids_mapping(probabilities):
    """Return the WNIDs sorted by probabilities."""
    categories = get_ilsvrc2012_WNIDs()
    check_input(probabilities)
    sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
    return np.take(categories, sorted_indices, axis=-1)
    
def imagnet_probabilities_to_1000class_names_mapping(probabilities):
    """Return the Class names sorted by probabilities."""
    categories = get_ilsvrc2012_categories()
    check_input(probabilities)
    sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
    return np.take(categories, sorted_indices, axis=-1)
    
def Imagenet_probabilities_to_16class_names_mapping(probabilities, aggregation_function=np.mean):
    """Return the 16 class categories sorted by probabilities"""
    aggregation_function = aggregation_function
    categories = get_human_object_recognition_categories()
    check_input(probabilities)
    aggregated_class_probabilities = []
    c = HumanCategories()
    for category in categories:
        indices = c.get_imagenet_indices_for_category(category)
        values = np.take(probabilities, indices, axis=-1)
        aggregated_value = aggregation_function(values, axis=-1)
        aggregated_class_probabilities.append(aggregated_value)
    aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
    sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
    return np.take(categories, sorted_indices, axis=-1)

