import shutil
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import api


class ApiClassNamesTest(unittest.TestCase):
    def test_dataset_class_names_follow_imagefolder_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for class_name in ["glass", "hazardous", "metal", "organic", "paper", "plastic", "textile"]:
                (Path(tmpdir) / class_name).mkdir(parents=True, exist_ok=True)

            class_names = api.get_class_names_from_data_path(tmpdir)

            self.assertEqual(class_names, ["glass", "hazardous", "metal", "organic", "paper", "plastic", "textile"])


if __name__ == "__main__":
    unittest.main()
