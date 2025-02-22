import os

from bids import BIDSLayout
from pathlib import Path

from connectomix.core.utils.tools import camel_case_list_of_strings, make_parent_dir


def add_new_entities(entities, config, label=None):

    match config["method"]:
        case "seedToVoxel":
            new_entity_key = "seed"
            new_entity_val = label
            suffix = "effectSize"
            entities["extension"] = "nii.gz"
        case "roiToVoxel":
            new_entity_key = "roi"
            new_entity_val = label
            suffix = "effectSize"
            entities["extension"] = "nii.gz"
        case "seedToSeed":
            new_entity_key = "data"
            new_entity_val = config["custom_seeds_name"]
            suffix = config["connectivity_kind"]
            entities["extension"] = "npy"
        case "roiToRoi":
            new_entity_key = "atlas"
            new_entity_val = config["atlas"]
            suffix = config["connectivity_kind"]
            entities["extension"] = "npy"
        case _:
            new_entity_key = None
            new_entity_val = None
            suffix = None

    entities["method"] = config["method"]
    entities["new_entity_key"] = new_entity_key
    entities["new_entity_val"] = new_entity_val
    entities["suffix"] = suffix

    entities["analysis_name"] = config.get("analysis_name", None)

    return entities


def setup_bidslayout(bids_dir, output_dir, derivatives=dict()):
    # Create derivative directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the dataset_description.json file
    from connectomix.core.utils.writers import write_dataset_description
    write_dataset_description(output_dir)

    # Create a BIDSLayout to parse the BIDS dataset and index also the derivatives
    return BIDSLayout(bids_dir, derivatives=[*list(derivatives.values()), output_dir])


def apply_nonbids_filter(entity, value, files):
    """
    Filter paths according to any type of entity, even if not allowed by BIDS.

    Parameters
    ----------
    entity : str
        The name of the entity to filter on (can be anything).
    value : str
        Entity value to filter.
    files : list
        List of paths to filters.

    Returns
    -------
    filtered_files : list
        List of paths after filtering is applied.

    """
    filtered_files = []
    if not entity == "suffix":
        entity = f"{entity}-"
    for file in files:
        if f"{entity}{value}" in os.path.basename(file).split("_"):
            filtered_files.append(file)
    return filtered_files


def remove_pair_making_entity(entities):
    """
    When performing paired tests, only one type of entity can be a list with 2 values (those are used to form pairs).
    This is the "pair making entity". This function sets this entity to None.

    Parameters
    ----------
    entities : dict
        Entities to be used to form pairs in paired test.

    Returns
    -------
    unique_entities : dict
        Same as entities, with one entity set to None if it was a list of length > 1 in the input.

    """
    # Note that this function has no effect on entities in the case of independent samples comparison or during regression analysis
    unique_entities = entities.copy()

    task = entities['task']
    run = entities['run']
    session = entities['session']

    if isinstance(task, list):
        if len(task) > 1:
            unique_entities['task'] = None
    if isinstance(run, list):
        if len(run) > 1:
            unique_entities['run'] = None
    if isinstance(session, list):
        if len(session) > 1:
            unique_entities['session'] = None

    return unique_entities


def build_output_path(layout, entities, label, level, config, **kwargs):
    entities = add_new_entities(entities, config, label)
    if kwargs:
        for key in kwargs.keys():
            entities[key] = kwargs[key]
    if level == "participant":
        pattern = "sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_[{new_entity_key}-][{new_entity_val}_]{suffix}.{extension}"
    elif level == "group":
        pattern = "group/{method}/{analysis_name}/[ses-{session}/][ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_[{new_entity_key}-][{new_entity_val}_][analysis-{analysis_name}_]{suffix}.{extension}"
    output_path = layout.derivatives.get_pipeline("connectomix").build_path(entities, path_patterns=[pattern], validate=False)
    make_parent_dir(output_path)

    return output_path


def alpha_value_to_bids_valid_string(alpha):
    alpha = str(alpha)
    return alpha.replace(".", "Dot")
