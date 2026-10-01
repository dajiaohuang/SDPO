from unittest.mock import Mock

import pytest

from verl.trainer.ppo.sdpo_utils import load_sdpo_teacher_checkpoint, save_sdpo_teacher_checkpoint


def test_save_sdpo_teacher_checkpoint_uses_dedicated_actor_subdirectory(tmp_path):
    manager = Mock()

    save_sdpo_teacher_checkpoint(manager, str(tmp_path), global_step=12, max_ckpt_to_keep=3)

    manager.save_checkpoint.assert_called_once_with(
        local_path=str(tmp_path / "sdpo_teacher"),
        hdfs_path=None,
        global_step=12,
        max_ckpt_to_keep=3,
    )


def test_load_sdpo_teacher_checkpoint_restores_dedicated_actor_subdirectory(tmp_path):
    manager = Mock()
    teacher_path = tmp_path / "sdpo_teacher"
    teacher_path.mkdir()

    load_sdpo_teacher_checkpoint(manager, str(tmp_path), del_local_after_load=True)

    manager.load_checkpoint.assert_called_once_with(
        local_path=str(teacher_path), hdfs_path=None, del_local_after_load=True
    )


def test_load_sdpo_teacher_checkpoint_rejects_legacy_actor_only_checkpoint(tmp_path):
    manager = Mock()

    with pytest.raises(FileNotFoundError, match="cannot be resumed faithfully"):
        load_sdpo_teacher_checkpoint(manager, str(tmp_path))

    manager.load_checkpoint.assert_not_called()


def test_disabled_sdpo_teacher_checkpoint_is_a_noop():
    save_sdpo_teacher_checkpoint(None, "unused", global_step=1)
    load_sdpo_teacher_checkpoint(None, "unused")
