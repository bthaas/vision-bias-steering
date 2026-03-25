"""
Evaluation captions for local steering sweep.

Edit CAPTIONS to swap captions in or out — run_local_sweep.py reads only this list.
Each entry needs:
  - "text":  the raw caption string
  - "label": "spatial" or "descriptive"

Captions 0–4 (indices) are the original five used in the coherence-frontier
experiments (Qwen-1.8B, layer 11) that produced the published results.
Captions 5–13 are the remaining handcrafted eval captions from
data/handcrafted_eval.json, included for broader coverage.
"""

CAPTIONS = [
    # -----------------------------------------------------------------------
    # Captions 0–4 — used in coherence-frontier experiments
    # -----------------------------------------------------------------------
    {
        "text": (
            "A lone hiker stands on top of a snow-dusted ridge looking out across "
            "a wide valley far below, with dark pine forests covering the slopes in "
            "the background and a narrow trail winding up from behind."
        ),
        "label": "spatial",
    },
    {
        "text": (
            "A bright orange and yellow maple tree stands beside a small dark pond, "
            "its round colorful leaves scattered across the ground on the near side."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A wide red double-decker bus is parked beside a row of tall narrow "
            "buildings, its bright yellow destination sign visible above the dark "
            "tinted windshield."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A tall iron gate stands at the far end of a narrow stone path, with a "
            "wooden bench placed directly in front of it and parallel rows of "
            "lampposts lining both sides."
        ),
        "label": "spatial",
    },
    {
        "text": (
            "A round wooden table sits at the center of a small kitchen, with copper "
            "pots hanging directly overhead from the ceiling, a narrow window on the "
            "far wall behind it, and a wooden stool tucked underneath."
        ),
        "label": "spatial",
    },
    # -----------------------------------------------------------------------
    # Captions 5–13 — remaining handcrafted eval captions
    # -----------------------------------------------------------------------
    {
        "text": (
            "A tall woman in a long blue dress stands in front of a white marble "
            "fountain, with a small brown suitcase sitting flat on the ground "
            "beside her."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A black cat perches on top of a rough gray chimney, staring down at a "
            "small yellow bird hopping on the bright green lawn far beneath."
        ),
        "label": "spatial",
    },
    {
        "text": (
            "A cracked terracotta pot sits at the base of a smooth white wall, with "
            "tiny bright green seedlings just visible above the rim and dry brown "
            "soil packed solid around the outside."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A narrow footbridge stretches horizontally across a wide dark river, "
            "its worn wooden planks visible below the thin metal railing, with a "
            "distant red lighthouse along the far bank."
        ),
        "label": "spatial",
    },
    {
        "text": (
            "A large dusty brown bookshelf packed with colorful worn volumes stands "
            "beside a tall frosted window, casting long shadows across the pale "
            "wooden floor beneath it."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A small red kayak floats at the center of a calm glassy lake, with tall "
            "dark pines surrounding the shore on all sides and a pale blue sky "
            "reflected in the still surface below."
        ),
        "label": "spatial",
    },
    {
        "text": (
            "An elderly man in a faded green jacket sits on a low wooden bench beside "
            "a busy street, a worn flat brown leather briefcase resting on the ground "
            "directly at his feet."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "Three weathered gray stone steps lead up toward a wide arched doorway "
            "painted deep blue, with an ornate black iron railing running along "
            "each side."
        ),
        "label": "descriptive",
    },
    {
        "text": (
            "A gleaming silver coffee pot sits at the center of a round white table "
            "beside an open window, with a small stack of worn brown books and a "
            "single yellow flower in a thin glass vase."
        ),
        "label": "descriptive",
    },
]
