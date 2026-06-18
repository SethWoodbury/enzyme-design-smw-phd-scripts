from utils.models import Interaction
from stages.stage1_activesite_remaster.run import _summarize_important_component


def test_summarize_component_defaults_to_sidechain():
    assert _summarize_important_component([]) == "sidechain"


def test_summarize_component_backbone():
    interactions = [
        Interaction(
            interaction_type="hbond_donation",
            from_component="backbone",
            from_atom="N",
            to_entity="ligand",
            to_atom="O1",
            distance=2.9,
        )
    ]
    assert _summarize_important_component(interactions) == "backbone"
