#!/usr/bin/env python3

# IMPORTS

from datetime import datetime
import os  # to interact with dirs (join paths etc)
import subprocess  # to call bin outside of python
from collections import defaultdict
import sys

# CLASSES

class Report:
    """
    A class to create, update, modify and save reports with pretty stuff
    """
    def __init__(self, path=None, string=""):
        """
        path is the place where the report is written/updated
        """
        self.path = path
        self.string = string

    def init(self, subject, date_and_time, version,
             cmd, session=None):
        """
        Init the report with html headers and various information on the report
        """
        with open(self.path, "w") as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<body>\n')
            f.write('<h1>Connectomix: individual report</h1>\n')
            f.write('<h2>Summary</h2>\n')
            f.write('<div>\n')
            f.write('<ul>\n')
            f.write(
                '<li>BIDS participant label: sub-%s</li>\n' % subject)
            if not session is None:
                f.write(
                    '<li>Session: ses-%s</li>\n' % session)
            f.write('<li>Date and time: %s</li>\n' % date_and_time)
            f.write('<li>Connectomix version: %s</li>\n' % version)
            f.write('<li>Command line options:\n')
            f.write('<pre>\n')
            f.write('<code>\n')
            f.write(str(cmd) + '\n')
            f.write('</code>\n')
            f.write('</pre>\n')
            f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</div>\n')

    def add_section(self, title):
        """
        Add a section to the report
        """
        with open(self.path, "a") as f:
            f.write('<h2>%s</h2>\n' % title)

    def add_subsection(self, title):
        """
        Add a subsection to the report
        """
        with open(self.path, "a") as f:
            f.write('<h3>%s</h3>\n' % title)

    def add_sentence(self, sentence):
        """
        Add a sentence to the report
        """
        with open(self.path, "a") as f:
            f.write('%s<br>\n' % sentence)

    def add_png(self, path):
        """
        Add an image to the report from a path to the png file
        """
        with open(self.path, "a") as f:
            f.write('<img width="400" src = "%s">' % path)

    def append(self, string):
        """
        Append string to report
        """
        with open(self.path, "a") as f:
            f.write(string)


    def finish(self):
        """
        Writes the last lines of the report to finish it.
        """
        with open(self.path, "a") as f:
            f.write('</body>\n')
            f.write('</html>\n')

# FUNCTIONS

def run(command, env={}):
    """Execute command as in a terminal

    Also prints any output of the command into the python shell
    Inputs:
        command: string
            the command (including arguments and options) to be executed
        env: to add stuff in the environment before running the command

    Returns:
        nothing
    """

    # Update env
    merged_env = os.environ
    merged_env.update(env)

    # Run command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)

    # Read whatever is printed by the command and print it into the
    # python console
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d" % process.returncode)


def arguments_manager(version):
    """
    Wrapper to define and read arguments for main function call
    Args:
        version: version to output when calling with -h

    Returns:
        args as read from command line call
    """
    import argparse

    parser = argparse.ArgumentParser(description = 'Entrypoint script.')
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
                        choices = ['participant',
                                   'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this parameter is not provided all subjects should be analyzed. Multiple participants can be specified with a space separated list.', nargs = "+")
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation', action='store_true')
    parser.add_argument('--fmriprep_dir', help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'restingstate\'.')
    parser.add_argument('--sessions', help='Name of the session to consider. If omitted, will loop over all sessions.', nargs = "+")
    parser.add_argument('--space', help='Name of the space to be used. Must be associated with fmriprep output.')
    parser.add_argument('--denoising_strategy', help='Names of the denoising strategy to consider. If omitted, set to \'simple\'. Examples are \'simple\', \'scrubbing\', \'compcor\', \'ica_aroma\'')
    parser.add_argument('--seeds_file',
                        help='Optional. Path to a .tsv file from which the nodes to compute the connectome will be loaded. The .tsv file should have four columns, without header. The first one contains the node name (a string) and the three others are the x,y,z coordinates in MNI space of the correspondings node. If omitted, it will load the msdl probabilitic atlas (see nilearn documentations for more info). ')
    parser.add_argument('-v', '--version', action='version', version='BIDS-App example version {}'.format(version))

    return parser.parse_args()


def get_space(args, layout):
    """
    Get space (and res, if any) and checks if present in layout (rawdata and derivatives)
    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for space entity
    """
    from .shellprints import msg_info, msg_error
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


def setup_subject_output_paths(output_dir, subject_label, space, res, session, strategy):
    """
    Setup various paths for subject output. Also creates subject output dir.
    Args:
        output_dir: str, cvrmap output dir
        subject_label: str, subject label
        space: str, space entity
        res: int or str, resolution entity
        session: str, session entity
        args: output of arguments_manager

    Returns:
        dict with various output paths (str)

    """
    from pathlib import Path  # to create dirs
    import os

    # create output_dir/sub-XXX directory

    ses_str = ''
    res_str = ''
    if session is not None:
        ses_str = 'ses-' + session
    if res is not None:
        res_str = '_res-' + res
    strategy_str = '_desc-' + strategy

    subject_output_dir = os.path.join(output_dir,
                                      "sub-" + subject_label, ses_str)
    ses_str = '_' + ses_str

    Path(subject_output_dir).mkdir(parents=True, exist_ok=True)

    # directory for figures
    figures_dir = os.path.join(subject_output_dir, 'figures')
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # directory for extras
    extras_dir = os.path.join(subject_output_dir, 'extras')
    Path(extras_dir).mkdir(parents=True, exist_ok=True)

    # set paths for various outputs
    outputs = {}

    subject_prefix = os.path.join(subject_output_dir,
                                      "sub-" + subject_label + ses_str)

    prefix = subject_prefix + "_space-" + space + res_str + strategy_str

    report_extension = '.html'
    data_extension = '.tsv'
    figures_extension = '.svg'

    # report is in root of derivatives (fmriprep-style), not in subject-specific directory

    outputs['report'] = os.path.join(output_dir,
                                     "sub-" + subject_label + '_report' + report_extension)

    # principal outputs
    outputs['data'] = prefix + '_data' + data_extension

    # supplementary data (extras)
    outputs['timeseries'] = os.path.join(extras_dir, "sub-" + subject_label
                                         + ses_str + "_space-" + space
                                         + res_str + strategy_str
                                         + '_timeseries' + data_extension)

    # figures (for the report)
    outputs['matrix'] = os.path.join(figures_dir, 'sub-' + subject_label
                                     + '_matrix' + figures_extension)
    outputs['connectome'] = os.path.join(figures_dir, 'sub-' + subject_label
                                         + '_connectome' + figures_extension)

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
    from .shellprints import msg_error
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


def get_fmriprep_dir(args):
    """
    Get and check existence of fmriprep dir from options or default
    Args:
        args: return from arguments_manager

    Returns:
        path to fmriprep dir
    """
    from os.path import join, isdir
    from .shellprints import msg_error
    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")
    # exists?
    if not isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)

    return fmriprep_dir


def get_sessions(args, layout):
    """
    Set the sessions to analyze. If argument "sessions" is specified, checks that they exist.
    Otherwise, select all available sessions in dataset.
    :param args: return from arguments_manager
    :param layout: BIDS layout
    :return: list of strings definind sessions to analyze
    """
    from .shellprints import msg_error
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
    from .shellprints import msg_error
    seeds = dict()
    if args.seeds_file:
        if not os.path.isfile(args.seeds_file):
            msg_error('Node file %s not found.' % args.seeds_file)
            sys.exit(1)
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
        seeds['type'] = 'msdl_atlas'
    return seeds

def get_strategy(args):
    """
    Get denoising strategies from options or set it to default
    :param args: return from arguments_manager
    :return: list of string, with strategies to apply
    """
    from .shellprints import msg_error
    available_strategies = ['simple', 'scrubbing', 'compcor',
                            'ica_aroma']
    # those are the four standard strategies directly available in nilearn
    if args.denoising_strategy:
        if args.denoising_strategy not in available_strategies:
            msg_error('Selected strategy %s is not available. Available strategies are %s.' % (args.denoising_strategy, available_strategies))
            sys.exit(1)
        strategies = args.denoising_strategy
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
    from .shellprints import msg_error
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
    matrix = plotting.plot_matrix(data, figure=(10, 8), labels=labels,
                                  vmax=0.8, vmin=-0.8, title=title,
                                  reorder=False)  # save with carpet.figure.savefig(path)
    connectome = plotting.plot_connectome(data, coords, title=title)  # save with connectome.savefig(path)
    #view = plotting.view_connectome(data, coords,
    #                                title=title)  # put in html by inserting the string view._repr_html_()

    plt.close()

    return dict(matrix=matrix.figure, connectome=connectome)

def get_masker_labels_coords(seeds):

    from nilearn import datasets
    from nilearn import plotting
    from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker

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

    return masker, labels, coordinates

def get_fmri_preproc(layout, bids_filter):
    """
    Selects preprocessing fmri file for
     current subject, space, resolution (if any), task and session
    :param layout: BIDSlayout
    :param bids_filter: dict
    :return: str, path to fmri file
    """
    from .shellprints import msg_error
    _filter = bids_filter.copy()
    _filter.update({'desc': "preproc"})
    fmri_files = layout.derivatives['fMRIPrep'].get(**bids_filter, desc='preproc', extension='.nii.gz')
    if len(fmri_files) == 1:
        fmri_file = fmri_files[0]
    else:
        msg_error('fmri file not found!')
        sys.exit(1)
    return fmri_file

def get_connectivity_measures(fmri_file, denoise_strategy, seeds):
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np

    _timeseries, _labels, _coords = get_timeseries(fmri_file, denoise_strategy, seeds)
    _correlation_matrix = ConnectivityMeasure(kind='correlation').fit_transform([_timeseries])[0]
    np.fill_diagonal(_correlation_matrix, 0)  # for visual purposes

    results = dict()
    results['data'] = _correlation_matrix
    results['graphics'] = get_graphics(_correlation_matrix, 'correlation' + ' - ' + denoise_strategy, _labels, _coords)
    results['timeseries'] = _timeseries

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

    options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}
    _confounds, _ = load_confounds_strategy(fmri_file, **options_for_load_confounds_strategy)

    _masker, _labels, _coords = get_masker_labels_coords(seeds)
    time_series = _masker.fit_transform(fmri_file, confounds=_confounds)

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
    from .shellprints import msg_error

    filter.update({'suffix': "bold"})
    if strategy == 'ica_aroma':
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

    if strategy == 'ica_aroma':
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
