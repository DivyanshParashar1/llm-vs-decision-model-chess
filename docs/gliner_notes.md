# GLiNER2.5-Decide: verified facts (smoke test, notebook 01, 2026-11-08)

Raw dump: `chess_data/gliner_notes.json` in the `chess-01-data` Kaggle output.

| Question (spec §2) | Answer |
|---|---|
| Package | `gliner2==2.1.0` (`pip install "gliner2[local]"`) |
| Load | `gliner2.AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")` → `SpanExtractor`, ~25 s on Kaggle CPU |
| Size | 486.4M params total (DeBERTa-v2/v3-large encoder + heads), not the advertised 340M |
| Classify call | `model.classify_text(text, {"best_move": labels}, include_confidence=True)` → `{'best_move': {'label': 'Qe2', 'confidence': 0.22}}` |
| Label descriptions | pass `{"best_move": {label: description, ...}}` |
| Batch | `model.batch_classify_text(texts, tasks, batch_size=8, include_confidence=True, format_results=True)` |
| All-label scores | `format_results=False` (format to be confirmed in notebook 02) |
| Max input | encoder `max_position_embeddings=512`, relative attention (256 buckets); 1,419-token inputs ran without error. Keep inputs ≤512 tokens to be safe |
| Labels per call | 5, 10, 20, 44 all OK (CPU: 1.1 s → 1.9 s). X1 (all legal moves) is feasible |
| Local fine-tuning | **yes**: `gliner2.training.trainer.GLiNER2Trainer(model, TrainingConfig(...), train_data=..., eval_data=...)` |
| Training example | `InputExample(text, classifications=[Classification(task="best_move", labels=[...], true_label="Nf3", label_descriptions={...})])` |
| TrainingConfig | `output_dir, num_epochs=10, max_steps, batch_size=2, eval_batch_size, gradient_accumulation_steps, encoder_lr=1e-5, task_lr=5e-4, weight_decay, ...` |
| LoRA | `LoRAConfig(enabled, r=8, alpha=16, dropout, target_modules)` available; full fine-tune is affordable on a T4 |

Zero-shot observation: with plain labels it picked `Qe2` in a mate-in-1 position; with label descriptions it picked `Qxf7#`.
