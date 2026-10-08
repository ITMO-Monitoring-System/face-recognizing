import os
import tempfile

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="face-tests-mpl-"))
