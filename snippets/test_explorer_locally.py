import warnings
warnings.filterwarnings("ignore")

import loop
from loop.explorer.loop_explorer import loop_explorer


def test_explorer_locally():

    print('Keep waiting, it will take 30 seconds...')

    historical = loop.HistoricalData();
    historical.get_spot_klines(kline_size=57600);
    uel = loop.UniversalExperimentLoop(data=historical.data, single_file_decoder=loop.sfd.foundational_sfd.logreg_binary)

    print('Keep waiting, there will now be three progress bars, once the last is done, you will see the streamlit url')

    uel.run(
            experiment_name='explorer_test',
            prep_each_round=True,
            n_permutations=100
        )

    loop_explorer(uel, '0.0.0.0')

# Install these locally to make Playwrite work
# python -m playwright install chromium
test_explorer_locally()