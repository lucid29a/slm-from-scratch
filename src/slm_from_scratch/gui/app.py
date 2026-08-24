"""A local Gradio app: chat with a checkpoint, and monitor a training run live.

Two tabs, both thin layers over existing pieces -- nothing here duplicates
model/training logic:

* **Chat** reuses :func:`slm_from_scratch.cli.commands.generate.generate`, the
  same KV-cached sampling loop ``slm generate`` uses.
* **Train** reuses :func:`slm_from_scratch.cli.commands.train.build_trainer`,
  the same config-driven builder ``slm train`` uses, running it on a
  background thread so the UI stays responsive, with a
  :class:`~slm_from_scratch.gui.callback.DashboardCallback` streaming metrics
  back to a :class:`~slm_from_scratch.gui.state.TrainingRunState` that a
  ``gr.Timer`` polls.

Because the base model here is never instruction-tuned (no SFT stage exists
yet -- see the project roadmap), "Chat" is honestly a free continuation of the
running conversation text, not a tuned assistant; the UI says so.
"""

from __future__ import annotations

import threading
import traceback

import gradio as gr
import torch

from slm_from_scratch.cli.commands.generate import generate as generate_text
from slm_from_scratch.cli.commands.train import build_trainer
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.gui.callback import DashboardCallback, DashboardSampleCallback
from slm_from_scratch.gui.state import TrainingRunState
from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.training.checkpoint import CheckpointManager

__all__ = ["build_app"]

_ChatModel = tuple[DecoderOnlyTransformer, Tokenizer, torch.device]


def _load_chat_model(
    model_config_path: str, checkpoint_path: str, tokenizer_path: str
) -> tuple[_ChatModel | None, str]:
    """Build a model, load a checkpoint into it, and load its tokenizer.

    Returns:
        ``(model_bundle, status_message)``; ``model_bundle`` is ``None`` on failure.
    """
    try:
        spec = load_yaml_mapping(model_config_path)
        model_config = ModelConfig.from_dict(spec["model"])
        model = DecoderOnlyTransformer(model_config)

        state = CheckpointManager(checkpoint_path).load_latest()
        if state is None:
            return None, f"No checkpoint found in {checkpoint_path}"
        model.load_state_dict(state.model_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        tokenizer = Tokenizer.load(tokenizer_path)
    except Exception as exc:
        return None, f"Failed to load: {exc}"

    params = model.num_parameters()
    return (model, tokenizer, device), f"Loaded ({params:,} non-embedding params) on {device}."


def _chat_respond(
    message: str,
    history: list[dict[str, str]],
    model_bundle: _ChatModel | None,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], str]:
    if model_bundle is None:
        history = [*history, {"role": "user", "content": message}]
        placeholder = "(No model loaded -- load one above first.)"
        history.append({"role": "assistant", "content": placeholder})
        return history, ""

    model, tokenizer, device = model_bundle
    # Base model, no chat template: the running transcript itself is the prompt.
    prompt = "".join(f"{turn['content']}\n" for turn in history if turn["role"] == "user")
    prompt += message

    reply = generate_text(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k or None,
        top_p=top_p,
        device=device,
    )
    continuation = reply[len(prompt) :].strip() or "(empty generation)"

    history = [*history, {"role": "user", "content": message}]
    history.append({"role": "assistant", "content": continuation})
    return history, ""


def build_app() -> gr.Blocks:
    """Build the Gradio app (call ``.launch()`` on the result to serve it).

    This is a single-user local tool, so training-run state is one module-level
    :class:`TrainingRunState` closed over by the handlers below, not per-session
    ``gr.State`` -- there is exactly one training run at a time regardless of
    how many browser tabs are open against it.
    """
    run_state = TrainingRunState()

    def start_training(config_path: str) -> str:
        if run_state.status == "running":
            return "A run is already in progress."

        try:
            spec = load_yaml_mapping(config_path)
        except Exception as exc:
            return f"Could not load config: {exc}"

        run_state.reset()
        run_state.status = "running"

        def worker() -> None:
            try:
                trainer = build_trainer(dict(spec))
                trainer.stop_event = run_state.stop_event
                trainer.callbacks.append(DashboardCallback(run_state))
                if spec.get("tokenizer") and spec.get("sample_every"):
                    tokenizer = Tokenizer.load(spec["tokenizer"])
                    trainer.callbacks.append(
                        DashboardSampleCallback(
                            run_state,
                            tokenizer,
                            prompt=spec.get("sample_prompt", "Once upon a time"),
                            every_n_steps=spec["sample_every"],
                        )
                    )
                trainer.train()
                if run_state.stop_event.is_set() and run_state.status == "running":
                    run_state.status = "stopped"
            except Exception as exc:
                run_state.status = "error"
                run_state.error = f"{exc}\n{traceback.format_exc()}"

        threading.Thread(target=worker, daemon=True).start()
        return f"Started training from {config_path}."

    def stop_training() -> str:
        if run_state.status != "running":
            return "No run in progress."
        run_state.status = "stopping"
        run_state.stop_event.set()
        return "Stop requested; finishing the current step."

    def poll_status() -> tuple[gr.LinePlot, str, str]:
        history = run_state.snapshot_history()
        plot_value = gr.LinePlot(
            value=[{"step": r.step, "loss": r.loss} for r in history], x="step", y="loss"
        )
        status_lines = [f"status: {run_state.status}"]
        if history:
            status_lines.append(f"step {history[-1].step}: loss={history[-1].loss:.4f}")
        if run_state.error:
            status_lines.append(f"error: {run_state.error}")
        return plot_value, "\n".join(status_lines), run_state.latest_sample

    with gr.Blocks(title="slm-from-scratch") as app:
        gr.Markdown("# slm-from-scratch")

        with gr.Tab("Chat"):
            gr.Markdown(
                "Base model, not instruction-tuned: this is free text continuation "
                "of the transcript below, not a tuned assistant."
            )
            with gr.Row():
                model_config_box = gr.Textbox(
                    label="Model config YAML", value="configs/model/modern_5m.yaml"
                )
                checkpoint_box = gr.Textbox(
                    label="Checkpoint dir", value="artifacts/cli_smoke/checkpoints"
                )
                tokenizer_box = gr.Textbox(
                    label="Tokenizer path", value="artifacts/cli_smoke/tokenizer.json"
                )
            load_button = gr.Button("Load model")
            load_status = gr.Textbox(label="Status", interactive=False)
            model_state = gr.State(None)

            chatbot = gr.Chatbot(height=400)
            msg_box = gr.Textbox(label="Message", placeholder="Once upon a time...")
            with gr.Row():
                temperature_slider = gr.Slider(0.0, 2.0, value=0.8, label="Temperature")
                top_k_slider = gr.Slider(0, 200, value=40, step=1, label="Top-k (0 = off)")
                top_p_slider = gr.Slider(0.0, 1.0, value=1.0, label="Top-p")
                max_new_slider = gr.Slider(1, 512, value=120, step=1, label="Max new tokens")

            load_button.click(
                _load_chat_model,
                inputs=[model_config_box, checkpoint_box, tokenizer_box],
                outputs=[model_state, load_status],
            )
            msg_box.submit(
                _chat_respond,
                inputs=[
                    msg_box,
                    chatbot,
                    model_state,
                    temperature_slider,
                    top_k_slider,
                    top_p_slider,
                    max_new_slider,
                ],
                outputs=[chatbot, msg_box],
            )

        with gr.Tab("Train"):
            gr.Markdown(
                "Runs the same config-driven builder as `slm train`, on a background thread."
            )
            train_config_box = gr.Textbox(
                label="Train config YAML", value="configs/train/smoke.yaml"
            )
            with gr.Row():
                start_button = gr.Button("Start training", variant="primary")
                stop_button = gr.Button("Stop")
            action_status = gr.Textbox(label="Action status", interactive=False)

            loss_plot = gr.LinePlot(x="step", y="loss", label="Training loss")
            run_status_box = gr.Textbox(label="Run status", interactive=False, lines=3)
            sample_box = gr.Textbox(label="Latest sample", interactive=False, lines=6)

            start_button.click(start_training, inputs=[train_config_box], outputs=[action_status])
            stop_button.click(stop_training, outputs=[action_status])

            timer = gr.Timer(2.0)
            timer.tick(poll_status, outputs=[loss_plot, run_status_box, sample_box])

    result: gr.Blocks = app
    return result
