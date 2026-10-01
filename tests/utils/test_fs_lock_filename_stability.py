import os
import subprocess
import sys


def test_local_mkdir_lock_filename_is_stable_across_processes():
    code = (
        "from verl.utils.fs import _get_local_mkdir_lock_filename; "
        "print(_get_local_mkdir_lock_filename('shared/checkpoint'))"
    )
    lock_filenames = []

    for hash_seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        lock_filenames.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                env=environment,
                text=True,
            ).strip()
        )

    assert lock_filenames[0] == lock_filenames[1]
