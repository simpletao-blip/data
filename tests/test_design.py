import numpy as np
import pandas as pd

from pca_ensemble.design import convex_hull_membership, farthest_point_indices, select_lbv_design


def test_farthest_point_selection_is_unique_and_deterministic():
    features = np.array([[0.0], [0.5], [1.0]])
    assert farthest_point_indices(features, 3) == [1, 0, 2]


def test_lbv_design_preserves_campaign_balance():
    rows = []
    for campaign in ["a", "b"]:
        for index in range(6):
            rows.append({"dataset_id": f"{campaign}{index}", "campaign_id": campaign,
                         "observable": "laminar burning velocity", "temperature_K": 300 + index,
                         "pressure_Pa": 1e5, "equivalence_ratio": 0.7 + index / 10,
                         "cracking_ratio": index / 10,
                         "initial_composition": '{"NH3": 0.1, "H2": 0.1, "O2": 0.2, "N2": 0.6}'})
    result = select_lbv_design(pd.DataFrame(rows), per_campaign=2)
    assert len(result) == 4
    assert result.groupby("campaign_id").size().to_dict() == {"a": 2, "b": 2}


def test_convex_hull_membership_2d():
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = np.array([[0.5, 0.5], [1.5, 0.5]])
    assert convex_hull_membership(reference, query).tolist() == [True, False]
