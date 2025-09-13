def test_explorer_locally():

    # Start Streamlit explorer for Logreg
    import warnings
    warnings.filterwarnings("ignore")

    print('Keep waiting, it will take 30 seconds...')

    import loop

    historical = loop.HistoricalData();
    historical.get_spot_klines(kline_size=57600);
    uel = loop.UniversalExperimentLoop(data=historical.data, single_file_model=loop.sfm.reference.logreg)

    print('Keep waiting, there will now be three progress bars, once the last is done, you will see the streamlit url')

    uel.run(experiment_name='explorer_test', prep_each_round=True, n_permutations=100)

    import sys; import importlib; sys.modules.pop('loop.explorer.loop_explorer', None); from loop.explorer.loop_explorer import loop_explorer; loop_explorer(uel, '0.0.0.0')


# Install these locally to make Playwrite work
# python -m playwright install chromium