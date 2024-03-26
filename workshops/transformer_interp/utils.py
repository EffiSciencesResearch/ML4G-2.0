"""
Code for plotting the logit lens, lightly adapted from
https://github.com/nostalgebraist/transformer-utils

MIT License

Copyright (c) 2021 nostalgebraist

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import partial
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns


def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)


def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark


def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)


def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)


def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)


def plot_logit_lens_low_level(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    probs=False,
    ranks=False,
    kl=False,
    top_down=False,
):
    end_ix = start_ix + layer_logits.shape[1]

    final_preds = layer_preds[-1]

    aligned_preds = layer_preds

    if kl:
        clip = 1 / (10 * layer_probs.shape[-1])
        final_probs = layer_probs[-1]
        to_show = kl_div(final_probs, layer_probs, clip=clip)
    else:
        numeric_input = layer_probs if probs else layer_logits

        to_show = get_value_at_preds(numeric_input, final_preds)

        if ranks:
            to_show = (numeric_input >= to_show[:, :, np.newaxis]).sum(axis=-1)

    _num2tok = np.vectorize(partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str])
    aligned_texts = _num2tok(aligned_preds)

    to_show = to_show[::-1]

    aligned_texts = aligned_texts[::-1]

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 0.375 * to_show.shape[0]))

    plot_kwargs = {"annot": aligned_texts, "fmt": ""}
    if kl:
        vmin, vmax = None, None

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
                "vmin": vmin,
                "vmax": vmax,
                "annot": True,
                "fmt": ".1f",
            }
        )
    elif ranks:
        vmax = 2000
        plot_kwargs.update(
            {
                "cmap": "Blues",
                "norm": matplotlib.colors.LogNorm(vmin=1, vmax=vmax),
                "annot": True,
            }
        )
    elif probs:
        plot_kwargs.update({"cmap": "Blues_r", "vmin": 0, "vmax": 1})
    else:
        vmin = np.percentile(to_show.reshape(-1), 5)
        vmax = np.percentile(to_show.reshape(-1), 95)

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40",
                "vmin": vmin,
                "vmax": vmax,
            }
        )

    sns.heatmap(to_show, **plot_kwargs)

    ax = plt.gca()
    input_tokens_str = _num2tok(input_ids[0].cpu())

    if layer_names is None:
        layer_names = ["Layer {}".format(n) for n in range(to_show.shape[0])]
    ylabels = layer_names[::-1]
    ax.set_yticklabels(ylabels, rotation=0)

    ax_top = ax.twiny()

    padw = 0.5 / to_show.shape[1]
    ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))

    ax_inputs = ax
    ax_targets = ax_top

    if top_down:
        ax.invert_yaxis()
        ax_inputs = ax_top
        ax_targets = ax

    ax_inputs.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    starred = [
        "* " + true if pred == true else " " + true
        for pred, true in zip(aligned_texts[0], input_tokens_str[start_ix + 1 : end_ix + 1])
    ]
    ax_targets.set_xticklabels(starred, rotation=0)
