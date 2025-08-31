def test_explorer_locally():

    # Start Streamlit explorer for Logreg
    import warnings
    warnings.filterwarnings("ignore")

    import loop

    historical = loop.HistoricalData();
    historical.get_spot_klines(kline_size=7200);
    uel = loop.UniversalExperimentLoop(data=historical.data, single_file_model=loop.sfm.reference.logreg)

    uel.run(experiment_name='explorer_test', prep_each_round=True, n_permutations=100)

    import sys; import importlib; sys.modules.pop('loop.explorer.loop_explorer', None); from loop.explorer.loop_explorer import loop_explorer; loop_explorer(uel, '0.0.0.0')
