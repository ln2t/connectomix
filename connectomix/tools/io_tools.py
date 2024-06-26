"""
Various input/output tools
"""

import sys
import os
from collections import defaultdict

def arguments_manager(version):
    """
        Wrapper to define and read arguments for main function call

    Args:
        version: version to output when calling with -h

    Returns:
        args as read from command line call
    """
    import argparse
    #  Deal with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('bids_dir', help='The directory with the input '
                                         'dataset formatted according to '
                                         'the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output '
                                           'files will be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that '
                                               'will be performed. '
                                               'Multiple participant level '
                                               'analyses can be run '
                                               'independently (in parallel)'
                                               ' using the same '
                                               'output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label',
                        help='The label(s) of the participant(s) that should be analyzed. '
                             'The label corresponds to sub-<participant_label> from the BIDS spec '
                             '(so it does not include "sub-"). If this parameter is not provided all subjects '
                             'should be analyzed. Multiple participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--fmriprep_dir',
                        help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'gas\'.')
    parser.add_argument('-v', '--version', action='version', version='cvrmap version {}'.format(version))
    parser.add_argument('--config',
                        help='Path to json file fixing the pipeline parameters. '
                             'If omitted, default values will be used.')
    return parser.parse_args()

def get_fmriprep_dir(args):
    """
        Get and check existence of fmriprep dir from options or default

    Args:
        args: return from arguments_manager

    Returns:
        path to fmriprep dir
    """
    from os.path import join, isdir
    from .shell_tools import msg_error
    import sys
    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")
    # exists?
    if not isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)
        sys.exit(1)

    return fmriprep_dir

def read_config_file(file=None):
    """
        Read config file

    Args:
        file, path to json file
    Returns:
        dict, with various values/np.arrays for the parameters used in main script
    """
    config = {}

    if file:
        import json
        from .shell_tools import msg_info

        msg_info('Reading parameters from user-provided configuration file %s' % file)

        with open(file, 'r') as f:
            config_json = json.load(f)

        keys = ['model', 'trials_type', 'contrasts',
                'tasks', 'first_level_options', 'concatenation_pairs']

        for key in keys:
            if key in config_json.keys():
                config[key] = config_json[key]

        config['first_level_options']['signal_scaling'] = tuple(config['first_level_options']['signal_scaling'])
    return config


def get_space(args, layout):
    """
    Get space (and res, if any) and checks if present in layout (rawdata and derivatives)
    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for space entity
    """
    from .shell_tools import msg_info, msg_error
    import sys
    if args.space:
        space = args.space
    else:
        space = 'MNI152NLin2009cAsym'
        msg_info('Defaulting to space %s' % space)

    # check if space arg has res modifier
    res = None
    if ":" in space:
        res = space.split(":")[1].split('-')[1]
        space = space.split(":")[0]

    # space in fmriprep output?
    spaces = layout.get_spaces(scope='derivatives')
    if space not in spaces:
        msg_error("Selected space %s is invalid. Valid spaces are %s" % (args.space, spaces))
        sys.exit(1)

    #todo: check if combination space+res is in fmriprep output

    return space, res


def setup_output_dir(args, version, layout):
    import os
    from pathlib import Path  # to create dirs
    # create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # initiate dataset_description file for outputs
    dataset_description = os.path.join(args.output_dir,
                                       'dataset_description.json')
    with open(dataset_description, 'w') as ds_desc:
        # todo: get the correct BIDS version
        ds_desc.write(('{"Name": "connectomix", "BIDSVersion": "x.x.x", '
                       '"DatasetType": "derivative", "GeneratedBy": '
                       '[{"Name": "connectomix"}, {"Version": "%s"}]}')
                      % version.rstrip('\n'))
        ds_desc.close()

    layout.add_derivatives(args.output_dir)  # add output dir as BIDS derivatives in layout

    return layout

def setup_subject_output_paths(layout, entities):
    """

    :param layout: BIDS layout
    :param entities: dict
    :return:
    """

    _layout = layout.derivatives['connectomix']
    outputs = {}

    patterns = dict()
    patterns['report'] = 'sub-{subject}[_ses-{session}]_desc-{denoising}_report.html'  # purposely at root
    patterns['data'] = 'sub-{subject}/sub-{subject}[_ses-{session}]_desc-{denoising}_data.tsv'
    patterns['timeseries'] = 'sub-{subject}/extras/sub-{subject}[_ses-{session}]_desc-{denoising}_timeseries.tsv'
    patterns['connectome'] = 'sub-{subject}/figures/sub-{subject}[_ses-{session}]_desc-{denoising}_connectome.svg'
    patterns['matrix'] = 'sub-{subject}/figures/sub-{subject}[_ses-{session}]_desc-{denoising}_matrix.svg'

    from os.path import dirname
    from pathlib import Path

    for item in ['report', 'data', 'timeseries', 'connectome', 'matrix']:
        outputs[item] = _layout.build_path(entities, patterns[item], validate=False)
        Path(dirname(outputs[item])).mkdir(parents=True, exist_ok=True)

    return outputs

def get_version ():
    """
    Print version from git info
    Returns:
    __version__
    """
    from os.path import join, dirname, realpath
    # version = open(join(dirname(realpath(__file__)), '..', '..', '..', '.git',
    #                        'HEAD')).read()
    version = open(join(dirname(realpath(__file__)), '..', '..', 'VERSION')).read()

    return version

def get_subjects_to_analyze(args, layout):
    """
    Generate list of subjects to analyze given the options and available subjects
    Args:
        args: return from arguments_manager
        layout: BIDS layout
    Returns:
        list of subjects to loop over
    """
    from .shell_tools import msg_error
    import sys
    if args.participant_label:  # only for a subset of subjects
        subjects_to_analyze = args.participant_label
        # in that case we must ensure the subjects exist in the layout:
        for subj in subjects_to_analyze:
            if subj not in layout.get_subjects():
                msg_error("Subject %s in not included in the "
                          "bids database." % subj)
                sys.exit(1)
    else:  # for all subjects
        subjects_to_analyze = sorted(layout.get_subjects())
    return subjects_to_analyze


def get_sessions(args, layout):
    """
    Set the sessions to analyze. If argument "sessions" is specified, checks that they exist.
    Otherwise, select all available sessions in dataset.
    :param args: return from arguments_manager
    :param layout: BIDS layout
    :return: list of strings definind sessions to analyze
    """
    from .shell_tools import msg_error
    sessions = [None]
    if args.sessions:
        for ses in args.sessions:
            if ses not in layout.get_sessions():
                msg_error("Selected session %s is not in the BIDS dataset. "
                          "Available session are %s." % (ses,
                                                       layout.get_sessions()))
                sys.exit(1)
        sessions = args.sessions
    else:
        if len(layout.get_sessions()) != 0:
            sessions =  layout.get_sessions()
    return sessions


def get_seeds(args):
    """
    Get the seeds to compute networks from.
    :param args: return from arguments_manager
    :return: dict defining the seeds with their type, labels, coordinates and radius.
    """
    import numpy as np
    import csv
    from .shell_tools import msg_error, msg_info
    seeds = dict()
    if args.seeds_file:
        if args.seeds_file == 'msdl':
            seeds['type'] = 'msdl_atlas'
            msg_info('Seeds set to regions of MSDL atlas')
        else:
            if not os.path.isfile(args.seeds_file):
                msg_error('Node file %s not found.' % args.seeds_file)
                sys.exit(1)
            else:
                seeds['type'] = 'customseeds'
                seeds['radius'] = 5
                with open(args.seeds_file) as file:
                    tsv_file = csv.reader(file, delimiter="\t")
                    seeds['labels'] = []
                    seeds['coordinates'] = []
                    for line in tsv_file:
                        seeds['labels'].append(line[0])
                        seeds['coordinates'].append(np.array(line[1:4], dtype=int))
                    # todo: add checks that what we are reading make sense
    else:
        seeds['type'] = 'all_voxels'
    return seeds

def get_fmri_mask(layout, bids_filter):
    """
    Get appropriate mask in fMRIPrep derivatives
    :param layout: BIDS layout
    :param bids_filter: dict
    :return: niimg for BOLD mask
    """
    from .shell_tools import msg_error
    from nilearn.image import load_img

    mask_files = layout.derivatives['fMRIPrep'].get(**bids_filter, desc='brain', suffix='mask', extension='.nii.gz')
    if len(mask_files) == 1:
        mask_file = mask_files[0]
        output = load_img(mask_file)
    else:
        msg_error('mask file not found!')
        output = None

    return output

def get_strategies(args):
    """
    Get denoising strategies from options or set it to default
    :param args: return from arguments_manager
    :return: list of string, with strategies to apply
    """
    from .shell_tools import msg_error
    available_strategies = ['simple', 'scrubbing', 'compcor', 'aroma', 'simpleGSR']
    # those are the four standard strategies directly available in nilearn
    if args.denoising_strategies:
        strategies = args.denoising_strategies
        for _strategy in args.denoising_strategies:
            if _strategy == 'all':
                strategies = available_strategies
            else:
                if _strategy not in available_strategies:
                    msg_error('Selected strategy %s is not available. Available strategies are %s.' % (args.denoising_strategies, available_strategies))
                    sys.exit(1)
    else:
        strategies = ['simple']

    return strategies

def get_task(args, layout):
    """
    Get and check task option or set to default
    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for task entity
    """
    from .shell_tools import msg_error
    import sys
    if args.task:
        task = args.task
    else:
        # fall back to default value
        task = "restingstate"

    if task not in layout.get_tasks():
        msg_error("Selected task %s is not in the BIDS dataset. "
                  "Available tasks are %s." % (args.task,
                                               layout.get_tasks()))
        sys.exit(1)

    return task


def nested_dict(n, type):
    """

    :param n: an integer defining the nesting level
    :param type: type of nested object, typically list or float
    :return: a nested dictionary
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))


def get_graphics(data, title, labels, coords):
    """

    :param data: numpy array with data
    :param title: title for the graphics
    :param labels: labels for the carpet plot
    :param coords: coordinates for the connectome graphs
    :return: dict with three fields:
                carpet: a matplotlib axes object
                connectome: a matplotlib figure object
                view: a nilearn view object
    """
    from nilearn import plotting
    from matplotlib import pyplot as plt

    if labels is None:
        labels = [' ' for i in range(data.shape[0])]
    matrix = plotting.plot_matrix(data, figure=(10, 8), labels=labels,
                                  title=title,
                                  vmin=-0.4,
                                  vmax=0.4,
                                  reorder=True)  # save with carpet.figure.savefig(path)

    if coords is None:
        connectome = None
    else:
        connectome = plotting.plot_connectome(data, coords, title=title)  # save with connectome.savefig(path)
    #view = plotting.view_connectome(data, coords,
    #                                title=title)  # put in html by inserting the string view._repr_html_()

    plt.close()

    return dict(matrix=matrix.figure, connectome=connectome)

def get_masker_labels_coords(seeds):

    from nilearn import datasets
    from nilearn import plotting
    from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker, NiftiMasker
    import numpy as np

    if seeds['type'] == 'msdl_atlas':
        # we load a probabilistic atlas
        dataset = datasets.fetch_atlas_msdl()
        masker = NiftiMapsMasker(maps_img=dataset.maps, standardize=True)
        labels = dataset.labels
        coordinates = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

        # for non-probabilistic atlases, use:
        # dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
        # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
        # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    if seeds['type'] == 'customseeds':
        masker = NiftiSpheresMasker(
            seeds['coordinates'], radius=seeds['radius'], detrend=True, standardize=True,
            low_pass=0.1, high_pass=0.01, t_r=2,
            memory='nilearn_cache', memory_level=1, verbose=2)
        labels = seeds['labels']
        coordinates = seeds['coordinates']

    if seeds['type'] == 'all_voxels':
        resampling_factor = 10
        masker = NiftiMasker(seeds['mask'], detrend=True, standardize=True,
                             target_affine=resampling_factor * np.identity(3))
        labels = None
        coordinates = None

    return masker, labels, coordinates

def get_fmri_preproc(layout, bids_filter, strategy):
    """
    Selects preprocessing fmri file for
     current subject, space, resolution (if any), task and session
    :param layout: BIDSlayout
    :param bids_filter: dict
    :return: str, path to fmri file
    """
    from .shell_tools import msg_error

    if strategy == 'aroma':
        desc = 'smoothAROMAnonaggr'
    else:
        desc = 'preproc'

    fmri_files = layout.derivatives['fMRIPrep'].get(**bids_filter, desc=desc, extension='.nii.gz')
    if len(fmri_files) == 1:
        fmri_file = fmri_files[0]
    else:
        msg_error('fmri file not found!')
        fmri_file = None
    return fmri_file

def get_connectivity_measures(fmri_file, denoise_strategy, seeds):
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np

    _timeseries, _labels, _coords = get_timeseries(fmri_file, denoise_strategy, seeds)
    kind = 'covariance'

    if _timeseries is not None:
        _correlation_matrix = ConnectivityMeasure(kind='covariance').fit_transform([_timeseries])[0]
        np.fill_diagonal(_correlation_matrix, 0)  # for visual purposes

        results = dict()
        results['data'] = _correlation_matrix
        results['graphics'] = get_graphics(_correlation_matrix, kind + ' - ' + denoise_strategy, _labels, _coords)
        results['timeseries'] = _timeseries
    else:
        results = None

    return results

def get_timeseries(fmri_file, strategy, seeds):
    """
    Compute connectomes and graphs of nodes for various denoising strategies.
    Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param fmri_file: str, path to fmri file to analyze
    :param strategy: str, setting the denoising strategy
    :param seeds: dict, to set regions to use as seeds. If seeds['type'] = 'msdl_atlas', will use probabilistic
    mdsl atlas. Otherwise: will read labels, coordinates and radius fields.
    :return: dict, with three fields: correlation, covariance and precision
    """

    from nilearn.interfaces.fmriprep import load_confounds_strategy
    from .shell_tools import msg_warning

    if strategy == 'aroma':
        options_for_load_confounds_strategy = {'denoise_strategy': 'ica_aroma'}
    else:
        if strategy == 'simpleGSR':
            options_for_load_confounds_strategy = {'denoise_strategy': 'simple', 'motion': 'full', 'global_signal': 'basic'}
        else:
            options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'full'}

    _confounds, _ = load_confounds_strategy(fmri_file, **options_for_load_confounds_strategy)

    _masker, _labels, _coords = get_masker_labels_coords(seeds)

    try:
        time_series = _masker.fit_transform(fmri_file, confounds=_confounds)
    except ValueError:
        msg_warning('There is an issue with this strategy, no confounds were loaded. Skipping.')
        time_series = None

    return time_series, _labels, _coords

def get_connectome(layout, filter, strategy, seeds):
    """
    Compute connectomes and graphs of nodes for various denoising strategies. Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param layout: BIDS layout containing the fmriprep outputs
    :param filter: dict to be used as a BIDS filter to select participants data
    :param strategy: str setting the denoising strategy
    :param seeds: dict to set regions to use as seeds. If seeds['type'] = 'msdl_atlas', will use probabilistic mdsl atlas. Other wise: will read labels, coordinates and radius fields.
    :return: dict with three fields: correlation, covariance and precision
    """

    # imports
    global correlation_matrix, caption, labels, coordinates, masker, fmri_files
    from nilearn import datasets
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np
    from nilearn import plotting
    from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker
    from .shell_tools import msg_error

    filter.update({'suffix': "bold"})
    if strategy == 'aroma':
        filter.update({'desc': "smoothAROMAnonaggr"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy}
    else:
        filter.update({'desc': "preproc"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}
    try:
        fmri_files = layout.get(**filter, extension='.nii.gz')[0]
    except:
        msg_error('fmri file %s not found' % fmri_files)
        sys.exit(1)
    confounds, sample_mask = load_confounds_strategy(fmri_files, **options_for_load_confounds_strategy)

    if seeds['type'] == 'msdl_atlas':

        # we load a probabilistic atlas
        dataset = datasets.fetch_atlas_msdl()
        masker = NiftiMapsMasker(maps_img=dataset.maps, standardize=True)
        labels = dataset.labels
        coordinates = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

        # for non-probabilistic atlases, use:
        # dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
        # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
        # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    if seeds['type'] == 'customseeds':
        masker = NiftiSpheresMasker(
            seeds['coordinates'], radius=seeds['radius'], detrend=True, standardize=True,
            low_pass=0.1, high_pass=0.01, t_r=2,
            memory='nilearn_cache', memory_level=1, verbose=2)
        labels = seeds['labels']
        coordinates = seeds['coordinates']

    try:
        time_series = masker.fit_transform(fmri_files, confounds=confounds)
    except:
        msg_error('There is a problem with series %s. Check QC of these data!' % fmri_files)
        sys.exit(1)

    correlation_matrix = ConnectivityMeasure(kind='correlation').fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)  # for visual purposes
    correlation_graphics = get_graphics(correlation_matrix, 'correlation' + ' - ' + strategy, labels, coordinates)

    return dict(type=seeds['type'], data=correlation_matrix, graphics=correlation_graphics)

def get_connectome_bk(layout, filter, strategy, nodes_coords, sphere_radius, nodes_labels):
    """
    Compute connectomes and graphs of nodes for various denoising strategies. Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param layout: BIDS layout containing the fmriprep outputs
    :param filter: dict() to be used as a BIDS filter to select participants data
    :param output_dir: directory where images will be saved
    :param report: report in which various figures and results will be saved. Must be initialized beforehand (report.init()).
    :param nodes: array of 3D coordinates for node
    :param radius: radius of the spheres centered on the nodes
    :return: dict with three fields: correlation, covariance and precision
    """

    # imports
    from nilearn import datasets
    from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np
    from nilearn import plotting
    from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker
    from sklearn.covariance import GraphicalLassoCV

    # atlas fetching

    # example of non-probabilistic atlases:
    #dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

    # we load a probabilistic atlas
    dataset = datasets.fetch_atlas_msdl()
    atlas_filename = dataset.maps
    labels = dataset.labels

    # for non-probabilistic atlases, use:
    # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
    # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
    atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

    nodes_labels = dataset.labels + nodes_labels
    nodes_coords = np.concatenate((plotting.find_probabilistic_atlas_cut_coords(dataset.maps), np.array(nodes_coords)))

    correlation_measure = ConnectivityMeasure(kind='correlation')

    filter.update({'suffix': "bold"})

    if strategy == 'aroma':
        # filter.update({'space': 'MNI152NLin6Asym'})
        filter.update({'desc': "smoothAROMAnonaggr"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy}
    else:
        filter.update({'desc': "preproc"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}

    # get the files
    fmri_files = layout.get(**filter, extension='.nii.gz')[0]

    # for non-probabilistic atlases, use:
    # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True)
    masker.fit(fmri_files)

    # we use pre-defined strategies
    confounds, sample_mask = load_confounds_strategy(fmri_files, **options_for_load_confounds_strategy)

    # extract now the time series with the current denoising strategy
    time_series = masker.fit_transform(fmri_files,
                                       confounds=confounds,
                                       sample_mask=sample_mask)

    # now that we have the time series, we can compute various stuff

    # correlation
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # for visual purposes we fill the diagonal with zeros
    np.fill_diagonal(correlation_matrix, 0)

    # covariance and precision
    # todo: understand the technical difference between the correlation matrix and the covariance matrix
    estimator = GraphicalLassoCV()
    estimator.fit(time_series)

    covariance_matrix = estimator.covariance_
    np.fill_diagonal(covariance_matrix, 0)

    precision_matrix = -estimator.precision_
    np.fill_diagonal(precision_matrix, 0)

    # for each of these measurements, we make two kinds of plots: a carpet-like plot and connectome plots (showing the nodes with their connection in the brain).
    # moreover, for the connectome, we can also make nice 3D dynamic figures (called a 'view' in nilearn).

    correlation = get_graphics(correlation_matrix, 'correlation' + ' - ' + strategy, labels, atlas_region_coords)
    covariance = get_graphics(covariance_matrix, 'covariance' + ' - ' + strategy, labels, atlas_region_coords)
    precision = get_graphics(precision_matrix, 'precision' + ' - ' + strategy, labels, atlas_region_coords)

    # now we compute stuff fot the list of nodes with ROI = spheres of radius sphere_radius

    masker_nodes = NiftiSpheresMasker(
        nodes_coords, radius=sphere_radius, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2,
        memory='nilearn_cache', memory_level=1, verbose=2)

    nodes_time_series = masker_nodes.fit_transform(fmri_files, confounds=[confounds])
    nodes_connectivity_measure = ConnectivityMeasure(kind='partial correlation')
    nodes_partial_correlation_matrix = nodes_connectivity_measure.fit_transform(
        [nodes_time_series])[0]

    nodes = get_graphics(nodes_partial_correlation_matrix, 'partial correlation' + ' - ' + strategy, nodes_labels, nodes_coords)

    from nilearn import plotting
    nodes_carpet = plotting.plot_matrix(nodes_partial_correlation_matrix, figure=(10, 8), labels=nodes_labels,
                                  vmax=0.8, vmin=-0.8,
                                  reorder=False)  # save with carpet.figure.savefig(path)

    return dict(correlation=correlation, covariance=covariance, precision=precision, nodes=nodes, nodes_plot=nodes_carpet, nodes_matrix=nodes_partial_correlation_matrix)
