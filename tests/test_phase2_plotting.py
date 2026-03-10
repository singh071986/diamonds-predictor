import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from Analyzer import Analyzer


class TestPhase2Plotting(unittest.TestCase):
    def setUp(self):
        self.sample_path = Path("tests/data/sample_diamonds.csv")
        self.output_dir = Path("tests/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_plot_functions_save_files(self):
        analyzer = Analyzer(str(self.sample_path))

        corr_path = self.output_dir / "corr.png"
        num_hist_path = self.output_dir / "num_hist.png"
        box_path = self.output_dir / "box_price.png"

        analyzer.plot_correlation_matrix(save_path=str(corr_path), show=False)
        analyzer.plot_histograms_numerical(save_path=str(num_hist_path), show=False)
        analyzer.plot_boxPlot("price", save_path=str(box_path), show=False)

        self.assertTrue(corr_path.exists())
        self.assertTrue(num_hist_path.exists())
        self.assertTrue(box_path.exists())


if __name__ == "__main__":
    unittest.main()
