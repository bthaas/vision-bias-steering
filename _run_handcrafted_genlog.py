"""
Standalone script: run generation logging on the handcrafted eval set.
Loads the gpt2 config + pre-trained steering vectors; sweeps specified lambdas.
"""
import json, torch, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from bias_steering.config import Config
from bias_steering.steering import load_model, get_target_token_ids
from bias_steering.data.load_dataset import load_handcrafted_eval
from bias_steering.eval.generation_logger import log_generations_sweep, build_output_path

# ── config ────────────────────────────────────────────────────────────────────

CFG_PATH       = "runs_vision/gpt2/config.yaml"
COEFFS         = [-150, -100, -50, 0, 50, 100, 150]
DECODING       = "greedy"
MAX_NEW_TOKENS = 20
OUTPUT_DIR     = Path("results/generation_logs")

cfg = Config.load(CFG_PATH)
layer = cfg.force_layer                      # 5 per saved config
INTERVENTION_METHOD = "default"              # stable for multi-token generation
CONSTRAINED_SOFTMAX = cfg.constrained_softmax  # True per GPT-2 config

logging.info("Config loaded: model=%s  layer=%d  constrained_softmax=%s",
             cfg.model_name, layer, CONSTRAINED_SOFTMAX)
logging.info("Using intervention_method=%s (overrides config's '%s' for generation)",
             INTERVENTION_METHOD, cfg.intervention_method)

# ── load model ────────────────────────────────────────────────────────────────

model = load_model(cfg.model_name)

# ── target token IDs ──────────────────────────────────────────────────────────

DATASET_DIR = Path("bias_steering/data/datasets")
target_words = json.load(open(DATASET_DIR / "target_words.json"))["vision"]

pos_ids_raw = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.pos_label])
neg_ids_raw = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.neg_label])

overlap = set(pos_ids_raw) & set(neg_ids_raw)
if overlap:
    logging.info("Removing %d overlapping token IDs", len(overlap))
pos_ids = [t for t in pos_ids_raw if t not in overlap]
neg_ids = [t for t in neg_ids_raw if t not in overlap]
logging.info("pos_ids: %d  neg_ids: %d", len(pos_ids), len(neg_ids))

# ── load steering vector ──────────────────────────────────────────────────────

artifact_dir = cfg.artifact_path()
candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
steering_vec = model.set_dtype(candidate_vectors[layer])
logging.info("Steering vector loaded: layer %d, shape %s", layer, steering_vec.shape)

# ── build examples from handcrafted eval ─────────────────────────────────────

hc_data = load_handcrafted_eval()
logging.info("Handcrafted examples: %d", len(hc_data))

examples = []
for _, row in hc_data.iterrows():
    caption = row["text"]
    raw_prompt = f"Describe this image:\n{caption}"
    if cfg.data_cfg.output_prefix:
        prompt = model.apply_chat_template([raw_prompt], output_prefix=["The image shows"])[0]
    else:
        prompt = model.apply_chat_template([raw_prompt])[0]
    examples.append({
        "example_id": int(row["_id"]),
        "caption": caption,
        "prompt_template": "image_shows",
        "prompt": prompt,
    })

# ── run generation sweep ──────────────────────────────────────────────────────

output_path = build_output_path(OUTPUT_DIR, "handcrafted_image_shows", DECODING)
logging.info("Output path: %s", output_path)

log_generations_sweep(
    model=model,
    examples=examples,
    layer=layer,
    steering_vec=steering_vec,
    coeffs=COEFFS,
    pos_ids=pos_ids,
    neg_ids=neg_ids,
    output_path=output_path,
    decoding=DECODING,
    max_new_tokens=MAX_NEW_TOKENS,
    intervention_method=INTERVENTION_METHOD,
    constrained_softmax=CONSTRAINED_SOFTMAX,
)
print(f"\nDone. Log written to: {output_path}")
print(f"Records: {len(examples)} examples × {len(COEFFS)} lambdas = {len(examples)*len(COEFFS)} total")
