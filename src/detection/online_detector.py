"""Online attention-ratio backdoor detector + trigger-candidate localizer.

Implements sliding-window (online) application of a trained logistic
detector over attention-ratio features (BackdoorFeatureExtractor).

Outputs window-level backdoor probabilities, raises alerts when
probabilities exceed a threshold for consecutive windows, and
produces token/sentence saliency scores plus an attention heatmap.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

from ..extraction import BackdoorFeatureExtractor
from ..detection import load_detector


class OnlineBackdoorDetector:
    def __init__(
        self,
        model,
        tokenizer,
        detector_path: str,
        device: str = "cuda",
        window_size: int = 8,
        stride: int = 1,
        threshold: float = 0.5,
        consecutive: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        self.consecutive = consecutive

        # Load extractor
        self.extractor = BackdoorFeatureExtractor(self.model, self.tokenizer, device=self.device)

        # Load classifier
        self.clf = load_detector(detector_path)

    def _clf_weights_abs(self, num_layers: int, num_heads: int) -> np.ndarray:
        """Return absolute weights reshaped to [num_layers, num_heads]."""
        coef = self.clf.coef_[0]  # [n_features]
        w = np.abs(coef)
        if w.size != num_layers * num_heads:
            w = np.resize(w, (num_layers * num_heads,))
        return w.reshape((num_layers, num_heads))

    def detect_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        top_k_sentences: int = 3,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run detection on a single prompt (simulate online with sliding windows).

        Returns a dict with window probabilities, alert windows, top sentences,
        and heatmap arrays (saved if output_dir provided).
        """
        # Extract generation attentions (collects all steps)
        generated_text, attn_data = self.extractor.extract_generation_attentions(
            prompt, max_new_tokens=max_new_tokens
        )

        # Create chunk features (sliding)
        chunk_feats = self.extractor.extract_all_chunks(
            attn_data, chunk_size=self.window_size, stride=self.stride
        )

        if len(chunk_feats) == 0:
            return {"error": "no chunks produced"}

        X = np.vstack([f.reshape(1, -1) for f in chunk_feats])

        # Compute probabilities for backdoor (1 = backdoor)
        probs = self.clf.predict_proba(X)[:, 1]

        # Find alerts where p > threshold for >= consecutive windows
        alert_windows = []
        consec = 0
        for i, p in enumerate(probs):
            if p > self.threshold:
                consec += 1
            else:
                consec = 0

            if consec >= self.consecutive:
                alert_windows.append(i)

        # Build head weights for saliency
        num_layers = attn_data.num_layers
        num_heads = attn_data.num_heads
        head_weights = self._clf_weights_abs(num_layers, num_heads)

        # Compute saliency S_i for last window (or each alert window)
        saliency_per_window = []  # list of [prompt_len] arrays
        heatmaps = []  # per-window heatmap arrays [window_len, prompt_len]

        prompt_len = len(attn_data.prompt_tokens)

        for win_idx in range(len(chunk_feats)):
            start_step = win_idx * self.stride
            end_step = min(start_step + self.window_size, len(attn_data.generated_tokens))

            # Token-level saliency: S_i = sum_{l,h} |w_{l,h}| * mean_t alpha^{l,h}_{t,i}
            S = np.zeros((prompt_len,), dtype=float)
            heat = np.zeros((end_step - start_step, prompt_len), dtype=float)

            for t_idx, step in enumerate(range(start_step, end_step)):
                # accumulate per-layer/head
                for layer in range(num_layers):
                    for head in range(num_heads):
                        w = head_weights[layer, head]
                        attn = attn_data.attentions[step, layer, 0, head, 0, :]
                        attn_np = attn.cpu().numpy()[:prompt_len]
                        S += w * attn_np
                        heat[t_idx] += w * attn_np

            # Average S across window steps
            S = S / max(1, (end_step - start_step))
            heat = heat  # per-step heat

            saliency_per_window.append(S)
            heatmaps.append(heat)

        # Sentence grouping: simple punctuation-based grouping using decoded tokens
        token_texts = [self.tokenizer.decode([int(t)]) for t in attn_data.prompt_tokens]
        sentences = []  # list of lists of token indices
        cur_sent = []
        for i, tok in enumerate(token_texts):
            cur_sent.append(i)
            if tok.strip().endswith(('.', '!', '?')):
                sentences.append(cur_sent)
                cur_sent = []
        if cur_sent:
            sentences.append(cur_sent)

        # Compute sentence scores for each window
        top_sentences_per_window = []
        for S in saliency_per_window:
            sent_scores = []
            for sent_idx_list in sentences:
                score = float(np.sum(S[sent_idx_list])) if len(sent_idx_list) > 0 else 0.0
                sent_scores.append(score)

            # Get top-k sentence indices
            if len(sent_scores) == 0:
                top_sentences_per_window.append([])
            else:
                idxs = np.argsort(sent_scores)[::-1][:top_k_sentences]
                top_sentences_per_window.append([(int(i), float(sent_scores[i])) for i in idxs])

        # Optionally save heatmaps / saliency
        out = {
            "generated_text": generated_text,
            "window_probs": probs.tolist(),
            "alert_windows": alert_windows,
            "saliency_per_window": [s.tolist() for s in saliency_per_window],
            "top_sentences_per_window": top_sentences_per_window,
        }

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            # Save arrays
            np.savez(
                os.path.join(output_dir, "online_detection_results.npz"),
                window_probs=probs,
                alert_windows=np.array(alert_windows, dtype=int),
            )

            # Save heatmaps as npy per window
            for i, heat in enumerate(heatmaps):
                np.save(os.path.join(output_dir, f"heatmap_window_{i}.npy"), heat)

        return out


__all__ = ["OnlineBackdoorDetector"]
