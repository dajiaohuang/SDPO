from verl.trainer.generation_utils import chats_to_object_array


def test_chats_to_object_array_preserves_ragged_conversations():
    chats = [
        [{"role": "user", "content": "question"}],
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    ]

    result = chats_to_object_array(chats)

    assert result.shape == (2,)
    assert result.tolist() == chats


def test_chats_to_object_array_preserves_equal_length_conversations_as_elements():
    chats = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]

    result = chats_to_object_array(chats)

    assert result.shape == (2,)
    assert result.tolist() == chats
