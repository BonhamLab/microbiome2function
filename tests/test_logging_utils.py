import logging
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath("src"))

from M2F.logging_utils import configure_logging


class TestLoggingUtils(unittest.TestCase):
    def setUp(self):
        self._root = logging.getLogger()
        self._old_handlers = list(self._root.handlers)
        self._old_level = self._root.level
        for h in list(self._root.handlers):
            self._root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        self.tmpdir = tempfile.mkdtemp(prefix="m2f_logs_")

    def tearDown(self):
        for h in list(self._root.handlers):
            self._root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        self._root.setLevel(self._old_level)
        for h in self._old_handlers:
            self._root.addHandler(h)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_configure_logging_sets_handlers_once(self):
        configure_logging(self.tmpdir, file_level=logging.INFO, console_level=logging.ERROR)
        first_count = len(self._root.handlers)
        self.assertGreaterEqual(first_count, 2)

        configure_logging(self.tmpdir, file_level=logging.DEBUG, console_level=logging.DEBUG)
        second_count = len(self._root.handlers)
        self.assertEqual(first_count, second_count)

        files = os.listdir(self.tmpdir)
        self.assertTrue(any(name.endswith(".log") for name in files))


if __name__ == "__main__":
    unittest.main()
