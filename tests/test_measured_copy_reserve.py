"""A failed reflink must not consume the checkpoint disk reserve."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from utils.measured_cases import copy_file


class CopyReserveTests(TestCase):
    def test_fallback_is_refused_without_copy_or_partial_target(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'source';source.write_bytes(b'unchanged source')
            target=Path(folder)/'target'
            with patch('fcntl.ioctl',side_effect=OSError('no reflink')),\
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=100)),\
                 patch('shutil.copy2',side_effect=AssertionError('unsafe fallback executed')):
                with self.assertRaises(OSError):copy_file(source,target,reserve_bytes=100)
            self.assertFalse(target.exists())
            self.assertEqual(source.read_bytes(),b'unchanged source')
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),['source'])

    def test_physical_fallback_preserves_source_and_uses_independent_inode(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'source';source.write_bytes(b'independent copy')
            target=Path(folder)/'target'
            with patch('fcntl.ioctl',side_effect=OSError('no reflink')):
                copy_file(source,target,reserve_bytes=100)
            self.assertEqual(source.read_bytes(),target.read_bytes())
            self.assertNotEqual(source.stat().st_ino,target.stat().st_ino)
