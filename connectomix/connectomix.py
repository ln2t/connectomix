#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""

# imports
from bids import BIDSLayout as bidslayout  # to handle BIDS data
from utils import *  # custom utilities
from pathlib import Path  # to create dirs

import numpy as np
from nilearn.connectome import ConnectivityMeasure

no_session_flag = 4.2  # used to for dummy value of session variable

def main():

    __version__ = get_version()
    msg_info("Version: %s" % __version__)

    args = arguments_manager(__version__)
    fmriprep_dir = get_fmriprep_dir(args)

    msg_info("Indexing BIDS dataset...")

    layout = bidslayout(args.bids_dir, validate=not args.skip_bids_validator)
    layout.add_derivatives(fmriprep_dir)
    subjects_to_analyze = get_subjects_to_analyze(args, layout)
    all_sessions = get_sessions(args, layout)
    space, res = get_space(args, layout)
    task = get_task(args, layout)
    seeds = get_seeds(args)
    denoise_strategies = get_strategies(args)
    layout = setup_output_dir(args, __version__, layout)
    output_dir = layout.derivatives['connectomix'].root

    # print some summary before running
    msg_info("Bids directory: %s" % layout.root)
    msg_info("Fmriprep directory: %s" % fmriprep_dir)
    msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    if not res is None:
        msg_info("Selected resolution: %s" % res)

    if args.analysis_level == "participant":

        for subject_label in subjects_to_analyze:

            msg_info("Running for participant %s" % subject_label)

            if subject_label not in layout.derivatives['fMRIPrep'].get_subjects():
                msg_warning('Subject to present in fMRIPrep derivatives, skipping.')
                continue

            sessions = []
            for session in all_sessions:
                if session in layout.get_sessions(subject=subject_label):
                    sessions.append(session)
                if session is None:
                    sessions = [None]
            sessions = sorted(sessions)
            if sessions[0] is not None:
                msg_info('Found the following sessions for this subject: %s' % sessions)

            for session in sessions:
                bids_filter = dict(subject=subject_label, return_type='filename',
                              space=space, res=res, task=task, session=session)

                if seeds['type'] == 'all_voxels':
                    seeds['mask'] = get_fmri_mask(layout, bids_filter)

                results = dict()

                for denoise_strategy in denoise_strategies:

                    msg_info('Running denoising strategy %s' % denoise_strategy)

                    # get fmri preprocessed and mask by fmriprep
                    fmri_preproc = get_fmri_preproc(layout, bids_filter, denoise_strategy)
                    if fmri_preproc is None:
                        continue

                    results[denoise_strategy] = get_connectivity_measures(fmri_preproc, denoise_strategy, seeds)

                    if results[denoise_strategy] is None:
                        continue

                    entities = dict()
                    entities['subject'] = subject_label
                    entities['space'] = space
                    entities['res'] = res
                    entities['session'] = session
                    entities['denoising'] = denoise_strategy
                    outputs = setup_subject_output_paths(layout, entities)

                    # save a couple of nice images
                    item = 'matrix'
                    results[denoise_strategy]['graphics'][item].savefig(outputs[item])

                    if not seeds['type'] == 'all_voxels':
                        item = 'connectome'
                        results[denoise_strategy]['graphics'][item].savefig(outputs[item])

                    # save the matrix data and timeseries
                    for item in ['data', 'timeseries']:
                        np.savetxt(outputs[item], results[denoise_strategy][item], delimiter='\t')

                    build_report(outputs, entities, args, __version__)

    # running group level
    elif args.analysis_level == "group":

        msg_info("Starting group analysis tools")

        import os

        group_level_dir = os.path.join(args.output_dir, 'group')

        from pathlib import Path  # to create dirs
        # create output dir
        Path(group_level_dir).mkdir(parents=True, exist_ok=True)

        import pandas as pd
        from plotly import graph_objects as go
        from os.path import join

        n_subjects = len(layout.derivatives['connectomix'].get_subjects())
        if n_subjects > 0:
            msg_info("Number of subject(s) found: %s" % n_subjects)
        else:
            msg_error("No subject found in derivative folder, abording.")
            exit(1)

        data_filter = dict(return_type='filename', extension='.nii.gz')
        strategies = layout.derivatives['connectomix'].get_desc()

        means = dict()
        for _strategy in strategies:
            means[_strategy] = []
            _list = layout.derivatives['connectomix'].get(return_type='filename', extension='.tsv', suffix='data', desc=_strategy)
            for _file in _list:
                _pd = pd.read_csv(_file, sep='\t', header=None)
                _data = _pd.values
                means[_strategy].append(np.mean(_data))

        subs = layout.derivatives['connectomix'].get_subjects()
        df = pd.DataFrame(columns=strategies)
        df['subject'] = []

        for _sub in subs:
            df.loc[len(df), 'subject'] = _sub
            for _strategy in strategies:
                _file = layout.derivatives['connectomix'].get(return_type='filename', subject=_sub, extension='.tsv', suffix='data', desc=_strategy)[0]
                _pd = pd.read_csv(_file, sep='\t', header=None)
                _data = _pd.values
                df.loc[df['subject'] == _sub, _strategy] = np.mean(_data)
        df.set_index('subject', inplace=True)

        table_group_filename = os.path.join(group_level_dir, "means_by_strategy.tsv")
        df.to_csv(table_group_filename, sep='\t')

        fig = go.Figure()

        for _strategy in strategies:
               fig.add_trace(go.Violin(y=df[_strategy],
                                    name=_strategy,
                                    box_visible=False,
                                    meanline_visible=False,
                                    points='all', showlegend=False))

        fig.update_layout(title="Mean Functional Connectivity by Denoising Strategy", yaxis_title="mean FC")
        # fig.show()

        fig_group_filename = os.path.join(group_level_dir, "denoising_compare.svg")
        fig.write_image(fig_group_filename)

        # time_series = dict()
        # bids_filter = dict(suffix='timeseries', extension='.tsv', return_type='filename')
        # msg_warning('Datasets without session not supported yet.')
        #
        # group_output_dir = os.path.join(output_dir, 'group')
        # Path(group_output_dir).mkdir(parents=True, exist_ok=True)
        #
        # for session in layout.derivatives['connectomix'].get_sessions():
        #     time_series[session] = dict()
        #     msg_warning('This tool should not be used if you have run several denoising strategies and the outputs are coexisting in the derivative folder.')
        #     for subject in layout.derivatives['connectomix'].get_subjects(session=session):
        #         try:
        #             series_fn = layout.derivatives['connectomix'].get(**bids_filter, subject=subject, session=session, desc=seeds['type'])[0]
        #         except:
        #             msg_warning('No series found for subject %s, skipping.' % subject)
        #             continue
        #         time_series[session][subject] = np.genfromtxt(series_fn, delimiter='\t')
        #
        #     time_series_stack = []
        #
        #     for key in time_series[session].keys():
        #         time_series_stack.append(time_series[session][key])
        #
        #
        #     measure = ConnectivityMeasure(kind='tangent')
        #     connectivities = measure.fit(time_series_stack)
        #     group_connectivity = measure.mean_
        #     data = measure.fit_transform(time_series_stack)[0]
        #     np.fill_diagonal(data, 0)  # for visual purposes
        #     graphics = get_graphics(data, 'Group connectome, ses-%s' % session, seeds['labels'], seeds['coordinates'])
        #     graphics['matrix']
        #
        #     group_prefix = os.path.join(group_output_dir, 'ses-%s' % session)
        #
        #     outputs_fn = dict()
        #     # save the graphics
        #     for item in ['matrix', 'connectome']:
        #         outputs_fn[item] = group_prefix + '_' + item + '.png'
        #         graphics[item].savefig(outputs_fn[item])
        #
        #     # save the matrix data and timeseries
        #     outputs_fn['data'] = group_prefix + '_data.tsv'
        #     np.savetxt(outputs_fn['data'], data, delimiter='\t')

    msg_info("The End!")

if __name__ == '__main__':
    main()
