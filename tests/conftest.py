import sys
from pathlib import Path

# Let the tests import `quantbot` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
