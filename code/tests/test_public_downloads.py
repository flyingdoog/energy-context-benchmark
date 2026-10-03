import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'scripts/download_sources.py'
SPEC = importlib.util.spec_from_file_location('public_downloads', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicDownloadsTest(unittest.TestCase):
    def test_default_is_offline_plan(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(MODULE.urllib.request, 'urlopen') as network:
            out = Path(tmp)/'not-created'
            with contextlib.redirect_stdout(io.StringIO()):
                plan = MODULE.main(['--source', 'cu_bems', '--output', str(out)])
            self.assertEqual(len(plan['files']), 5)
            self.assertFalse(out.exists())
            network.assert_not_called()

    def test_bpa_seasons(self):
        self.assertEqual([x['name'] for x in MODULE.source_plan('bpa', [])['files']],
                         ['winter2018.zip', 'spring2018.zip', 'summer2018.zip'])

    def test_download_and_refuse_overwrite(self):
        plan = dict(files=[dict(name='fixture.csv', url='https://example.org/fixture', size=3)])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(MODULE.urllib.request, 'urlopen', return_value=io.BytesIO(b'abc')):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.download_files(plan, Path(tmp))
            self.assertEqual((Path(tmp)/'fixture.csv').read_bytes(), b'abc')
            with self.assertRaises(FileExistsError):
                MODULE.download_files(plan, Path(tmp))

    def test_short_download_is_not_published(self):
        plan = dict(files=[dict(name='fixture.csv', url='https://example.org/fixture', size=5)])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(MODULE.urllib.request, 'urlopen', return_value=io.BytesIO(b'abc')):
                with self.assertRaises(ValueError):
                    MODULE.download_files(plan, Path(tmp))
            self.assertFalse((Path(tmp)/'fixture.csv').exists())


if __name__ == '__main__':
    unittest.main()
