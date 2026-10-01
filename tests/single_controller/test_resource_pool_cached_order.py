from unittest.mock import patch

from verl.single_controller.ray.base import RayResourcePool


class FakePlacementGroup:
    def ready(self):
        return None


def test_resource_pool_returns_sorted_placement_groups_on_first_call():
    first, second = FakePlacementGroup(), FakePlacementGroup()
    sorted_groups = [second, first]
    pool = RayResourcePool(process_on_nodes=[1, 1], use_gpu=False)

    with (
        patch("verl.single_controller.ray.base.placement_group", side_effect=[first, second]),
        patch("verl.single_controller.ray.base.ray.get"),
        patch("verl.single_controller.ray.base.sort_placement_group_by_node_ip", return_value=sorted_groups),
    ):
        initial_result = pool.get_placement_groups()
        cached_result = pool.get_placement_groups()

    assert initial_result == sorted_groups
    assert cached_result == sorted_groups
