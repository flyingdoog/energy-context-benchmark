"""The public model launcher works without a globally installed project API."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_model.py'


class ModelLauncherTests(unittest.TestCase):
    def test_import_arguments_and_sibling_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'user_model.py'
            (Path(tmp) / 'helper.py').write_text('VALUE = 17\n')
            script.write_text(
                'import json, sys\n'
                'from field_control import OperationalForecastTask\n'
                'from helper import VALUE\n'
                'print(json.dumps(dict(api=OperationalForecastTask.__name__, '
                'args=sys.argv[1:], sibling=VALUE, heavy=[name for name in '
                '["torch", "lightgbm", "sklearn"] if name in sys.modules])))\n')
            result = subprocess.run([sys.executable, str(LAUNCHER), str(script),
                                     '--width', '10', 'with spaces'], cwd=tmp,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = json.loads(result.stdout)
            self.assertEqual(result['api'], 'OperationalForecastTask')
            self.assertEqual(result['args'], ['--width', '10', 'with spaces'])
            self.assertEqual(result['sibling'], 17)
            self.assertEqual(result['heavy'], [])

    def test_missing_file_has_useful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(LAUNCHER),
                                     str(Path(tmp) / 'missing.py')], cwd=tmp,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Model script does not exist', result.stderr)


if __name__ == '__main__':
    unittest.main()
