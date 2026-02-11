"""Run token steering demo for Qwen and open plot in browser."""
import sys
sys.path.insert(0, '..')
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from bias_steering.steering import load_model, get_intervention_func
from bias_steering.utils import loop_coeffs

torch.set_grad_enabled(False)

# --- CONFIG ---
MODEL_NAME = 'Qwen/Qwen-1_8B-chat'
MODEL_ALIAS = 'Qwen-1_8B-chat'
LAYER = 11

print(f'Loading {MODEL_NAME}...')
model = load_model(MODEL_NAME)
artifact_dir = Path(f'../runs_vision/{MODEL_ALIAS}')
candidate_vectors = torch.load(artifact_dir / 'activations/candidate_vectors.pt')
steering_vec = model.set_dtype(candidate_vectors[LAYER])
print(f'Loaded, steering vec shape: {steering_vec.shape}')

# Tokens to track
spatial_tokens = [' left', ' right', ' behind', ' near', ' above', ' below', ' next']
descriptive_tokens = [' red', ' blue', ' large', ' small', ' round', ' bright', ' dark']
all_tokens = spatial_tokens + descriptive_tokens
token_ids = [model.tokenizer.encode(t, add_special_tokens=False)[0] for t in all_tokens]
token_labels = [t.strip() for t in all_tokens]
print('Tracking tokens:', token_labels)

# Prompt
caption = 'A cat sitting on a mat in a room with furniture.'
prompt = f'Continue describing this scene:\n{caption}'
p = model.apply_chat_template([prompt])[0]
p += 'The scene is'
print(f'Prompt: {p[:80]}...')

# Sweep
coeffs = list(loop_coeffs(min_coeff=-80, max_coeff=80, increment=10))
token_probs = {label: [] for label in token_labels}

for coeff in coeffs:
    print(f'  coeff={coeff:.0f}', end='', flush=True)
    intervene_func = get_intervention_func(steering_vec, method='constant', coeff=coeff)
    logits = model.get_logits([p], layer=LAYER, intervene_func=intervene_func)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    for tid, label in zip(token_ids, token_labels):
        token_probs[label].append(probs[tid].item())
print('\nDone sweeping!')

# Plot
colors_spatial = px.colors.qualitative.Set1[:len(spatial_tokens)]
colors_desc = px.colors.qualitative.Set2[:len(descriptive_tokens)]
fig = go.Figure()

for i, label in enumerate([t.strip() for t in spatial_tokens]):
    fig.add_trace(go.Scatter(
        x=coeffs, y=token_probs[label],
        mode='lines+markers', name=f'{label} (spatial)',
        marker_color=colors_spatial[i], line=dict(width=2),
    ))
for i, label in enumerate([t.strip() for t in descriptive_tokens]):
    fig.add_trace(go.Scatter(
        x=coeffs, y=token_probs[label],
        mode='lines+markers', name=f'{label} (desc)',
        marker_color=colors_desc[i], line=dict(width=2, dash='dash'),
    ))

fig.add_vline(x=0, line_dash='solid', line_color='black', line_width=1)
fig.update_layout(
    title=f'Token Probabilities vs Steering Coefficient ({MODEL_ALIAS}, Layer {LAYER})',
    title_font=dict(size=16), title_x=0.5,
    width=850, height=500, plot_bgcolor='white',
    legend=dict(title='Token', font=dict(size=13)),
)
fig.update_xaxes(title='Steering Coefficient (\u03bb)', showgrid=True, gridcolor='lightgrey')
fig.update_yaxes(title='Probability', showgrid=True, gridcolor='lightgrey', range=[0, None])

# Save and open in browser
out_path = Path('../plots/qwen_token_steering.html')
out_path.parent.mkdir(exist_ok=True)
fig.write_html(str(out_path), auto_open=True)
print(f'Saved to {out_path.resolve()} and opened in browser')
