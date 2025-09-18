from loop.features.conserved_flux_renormalization import conserved_flux_renormalization
from loop.tests.utils.get_data import get_trades_data


def test_conserved_flux_renormalization():

    trades_df = get_trades_data()

    _ = conserved_flux_renormalization(trades_df)
