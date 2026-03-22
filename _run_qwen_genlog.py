"""
Standalone script: run generation logging on the handcrafted eval set — Qwen-1.8B-chat.
"""
import json, torch, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from bias_steering.config import Config
from bias_steering.steering import load_model, get_target_token_ids
from bias_steering.data.load_dataset import load_handcrafted_eval
from bias_steering.eval.generation_logger import log_generations_sweep, build_output_path

CFG_PATH       = "runs_vision/Qwen-1_8B-chat/config.yaml"
COEFFS         = [-150, -100, -50, 0, 50, 100, 150]
DECODING       = "greedy"
MAX_NEW_TOKENS = 20
OUTPUT_DIR     = Path("results/generation_logs")

cfg = Config.load(CFG_PATH)

# Best layer by RMSE from top_layers.json
top_layers = json.load(open(cfg.artifact_path() / "validation/top_layers.json"))
layer = top_layers[0]["layer"]  # layer 12

INTERVENTION_METHOD = "default"
CONSTRAINED_SOFTMAX = cfg.constrained_softmax  # False for Qwen (not in yaml → default)

logging.info("Config loaded: model=%s  layer=%d  use_offset=%s  constrained_softmax=%s",
             cfg.model_name, layer, cfg.use_offset, CONSTRAINED_SOFTMAX)

model = load_model(cfg.model_name)

# Neutral offset (Qwen has use_offset=True)
offset = 0
if cfg.use_offset:
    artifact_dir = cfg.artifact_path()
    neutral_acts = torch.load(artifact_dir / "activations/neutral.pt")
    offset = model.set_dtype(neutral_acts.mean(dim=1)[layer])
    logging.info("Loaded neutral offset for layer %d, shape %s", layer, offset.shape)

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

candidate_vectors = torch.load(cfg.artifact_path() / "activations/candidate_vectors.pt")
steering_vec = model.set_dtype(candidate_vectors[layer])

hc_data = load_handcrafted_eval()
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

output_path = build_output_path(OUTPUT_DIR, "qwen_handcrafted_image_shows", DECODING)

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
    offset=offset,
)
print(f"\nDone. Log written to: {output_path}")
