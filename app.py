"""FoodLens: upload a food photo, get the dish name and a nutrition estimate.

Run with:  streamlit run app.py
Trained checkpoints (.pt files saved by the notebook) go in ./checkpoints.
"""
from __future__ import annotations

import html
import io
import os
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from foodlens.data import build_transforms, pretty_label
from foodlens.models import DISPLAY_NAMES, build_model
from foodlens.nutrition import DAILY_KCAL, PORTIONS, energy_split, estimate, load_table

CHECKPOINT_DIR = Path(os.environ.get("FOODLENS_CHECKPOINTS", Path(__file__).parent / "checkpoints"))
DEFAULT_CHECKPOINT = "best.pt"
SURE, LIKELY = 0.70, 0.40  # confidence bands shown to the user
EVAL_TF = build_transforms(train=False)
MACRO_COLORS = {"protein": "#8C4A3B", "carbs": "#D9A441", "fat": "#7A8B5C"}
OTHER = "other"

st.set_page_config(page_title="FoodLens", page_icon=":material/restaurant:", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3.5rem; max-width: 780px;}
    .fl-title {font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.6rem;
               letter-spacing: -0.015em; line-height: 1.05; color: #221E1A;}
    .fl-sub {color: #6B6158; font-size: 1.05rem; margin: .4rem 0 1.5rem;}
    .fl-result {margin-bottom: 1.25rem;}
    .fl-kicker {color: #8A7F74; font-size: .8rem; letter-spacing: .06em; text-transform: uppercase;}
    .fl-dish {font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.1rem;
              line-height: 1.12; color: #221E1A; margin: .15rem 0 .4rem;}
    .fl-note {color: #5E554D; margin-bottom: 1.1rem;}
    .fl-row {display: flex; justify-content: space-between; gap: 1rem; font-size: .95rem;
             margin-top: .7rem; color: #3A332D;}
    .fl-row span:last-child {color: #8A7F74; font-variant-numeric: tabular-nums;}
    .fl-bar {height: 6px; background: #ECE4D8; border-radius: 3px; margin-top: .3rem; overflow: hidden;}
    .fl-bar i {display: block; height: 100%; background: #B5442C; border-radius: 3px;}
    .fl-bar.alt i {background: #CDB9A5;}
    .fl-section {display: flex; align-items: baseline; gap: .6rem; margin: 2rem 0 .9rem;
                 padding-top: 1.4rem; border-top: 1px solid #E4DACB;}
    .fl-h {font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.45rem; color: #221E1A;}
    .fl-card {background: #F2ECE3; border-radius: 10px; padding: 1.2rem 1.25rem 1.1rem;}
    .fl-kcal {font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.6rem;
              line-height: 1; color: #221E1A; font-variant-numeric: tabular-nums;}
    .fl-kcal span {font-family: inherit; font-size: 1rem; font-weight: 500; color: #6B6158; margin-left: .35rem;}
    .fl-serving {color: #6B6158; font-size: .9rem; margin-top: .45rem;}
    .fl-split {display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 1.1rem 0 .85rem;
               background: #E4DACB;}
    .fl-split i {display: block; height: 100%;}
    .fl-macros {display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem;}
    .fl-lbl {font-size: .8rem; color: #6B6158; display: flex; align-items: center; gap: .4rem;}
    .fl-lbl i {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
    .fl-val {font-size: 1.2rem; font-weight: 600; color: #221E1A; font-variant-numeric: tabular-nums;}
    .fl-micro {font-size: .85rem; color: #6B6158; margin-top: .9rem; padding-top: .75rem;
               border-top: 1px solid #E4DACB; line-height: 1.6;}
    .fl-fine {color: #8A7F74; font-size: .8rem; margin: .6rem 0 1.2rem;}
    [data-testid="stButtonGroup"] > div {flex-wrap: wrap; overflow-x: visible; row-gap: .45rem;}
    .fl-foot {color: #8A7F74; font-size: .85rem; margin-top: 2.5rem;}
    .fl-list {columns: 3 10rem; font-size: .85rem; color: #3A332D; line-height: 1.6;}
    [data-testid="stImage"] img {border-radius: 8px; width: 100%; max-height: 420px; object-fit: cover;}
    .fl-total-m {display: none;}
    @media (max-width: 640px) {
      .block-container {padding-top: 2rem;}
      .fl-title {font-size: 2.2rem;}
      [data-testid="stImage"] img {max-height: 240px;}
      .fl-fine {margin-bottom: 0;}
      .fl-total-m {display: block; background: #F2ECE3; border-radius: 8px; padding: .6rem .9rem;
                   color: #6B6158; margin-top: .25rem;}
      .fl-total-m b {font-family: 'Fraunces', Georgia, serif; color: #221E1A; font-size: 1.15rem; margin-left: .3rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label = DISPLAY_NAMES.get(ckpt["arch"], ckpt["arch"])
    if ckpt.get("experiment"):
        label += f" ({ckpt['experiment']})"
    return model, list(ckpt["class_names"]), label


def read_image(source) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(source))  # phone photos keep rotation in EXIF
    image.thumbnail((1280, 1280))
    return image.convert("RGB")


@st.cache_data(show_spinner=False, max_entries=64)
def predict(image_bytes: bytes, model_path: str, k: int = 5):
    """Top-k (class index, probability) and inference time; cached so tweaking nutrition is instant."""
    model = load_checkpoint(model_path)[0]
    x = EVAL_TF(read_image(io.BytesIO(image_bytes))).unsqueeze(0)
    start = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)], elapsed_ms


def percent(p: float) -> str:
    return "<1%" if p < 0.01 else f"{p:.0%}"


def bar(name: str, p: float, alt: bool = False) -> str:
    return (f'<div class="fl-row"><span>{html.escape(name)}</span><span>{percent(p)}</span></div>'
            f'<div class="fl-bar{" alt" if alt else ""}"><i style="width:{max(p, 0.01) * 100:.1f}%"></i></div>')


def result_html(top, class_names) -> str:
    names = [pretty_label(class_names[i]) for i, _ in top]
    best = top[0][1]
    alternatives = [j for j in (1, 2) if top[j][1] >= 0.05]  # hide near-zero guesses
    if best >= SURE:
        kicker, note, rows = "This looks like", "Pretty sure about this one.", [0]
    elif best >= LIKELY:
        kicker, note, rows = "Probably", f"Could also be {names[1].lower()}.", [0, *alternatives]
    else:
        kicker, note, rows = "Hard to tell. Best guess", "Try a closer, well-lit photo of one dish.", [0, *alternatives]
    bars = "".join(bar(names[j], top[j][1], alt=j > 0) for j in rows)
    return (f'<div class="fl-result"><div class="fl-kicker">{kicker}</div>'
            f'<div class="fl-dish">{html.escape(names[0])}</div>'
            f'<div class="fl-note">{note}</div>{bars}</div>')


def nutrition_html(dish: str, factor: float, total: dict, table: dict) -> str:
    info = table["dishes"][dish]
    serving = f"{pretty_label(dish)} · {info['serving'].lower()}"
    if factor != 1:
        serving += f" × {factor:g}"
    serving += f", about {info['grams'] * factor:,.0f} g"
    split = energy_split(total)
    bars = "".join(f'<i style="width:{split[k] * 100:.1f}%;background:{c}"></i>' for k, c in MACRO_COLORS.items())
    macros = "".join(
        f'<div><div class="fl-lbl"><i style="background:{MACRO_COLORS[k]}"></i>{k.capitalize()}</div>'
        f'<div class="fl-val">{total[k + "_g"]:.0f} g</div></div>' for k in MACRO_COLORS)
    day = total["kcal"] / DAILY_KCAL
    return (f'<div class="fl-card"><div class="fl-kcal">{round(total["kcal"], -1):,.0f}<span>kcal</span></div>'
            f'<div class="fl-serving">{html.escape(serving)}</div>'
            f'<div class="fl-split">{bars}</div><div class="fl-macros">{macros}</div>'
            f'<div class="fl-micro">Fiber {total["fiber_g"]:.0f} g · Sugar {total["sugar_g"]:.0f} g · '
            f'Sodium {round(total["sodium_mg"], -1):,.0f} mg<br>About {day:.0%} of a {DAILY_KCAL:,} kcal day</div></div>'
            f'<div class="fl-fine">Estimated from USDA food data for a typical serving. Real meals vary.</div>')


st.markdown('<div class="fl-title">FoodLens</div>'
            '<div class="fl-sub">Upload a photo of a dish to see what it is and roughly what\'s in it.</div>',
            unsafe_allow_html=True)

checkpoints = sorted(str(p) for p in CHECKPOINT_DIR.glob("*.pt"))
if not checkpoints:
    st.info(f"No trained model yet. Put a checkpoint (.pt) from the notebook in `{CHECKPOINT_DIR}`.")
    st.stop()

default = next((p for p in checkpoints if Path(p).name == DEFAULT_CHECKPOINT), checkpoints[0])
if st.session_state.get("model_path") not in checkpoints:
    st.session_state["model_path"] = default

upload = st.file_uploader("Food photo", type=["jpg", "jpeg", "png", "webp"],
                          label_visibility="collapsed")
model_path = st.session_state["model_path"]
_, class_names, model_label = load_checkpoint(model_path)
table = load_table()

if upload is not None:
    image_bytes = upload.getvalue()
    try:
        image = read_image(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError):
        st.error("That file couldn't be opened as a photo. Try a JPG or PNG.")
        st.stop()

    top, elapsed_ms = predict(image_bytes, model_path)
    left, right = st.columns(2, gap="large")
    left.image(image)
    right.markdown(result_html(top, class_names), unsafe_allow_html=True)

    predicted = class_names[top[0][0]]
    dishes = sorted((d for d in class_names if d in table["dishes"]), key=pretty_label)
    if dishes:
        st.markdown('<div class="fl-section"><span class="fl-h">Nutrition</span>'
                    '<span class="fl-kicker">estimate</span></div>', unsafe_allow_html=True)
        card, adjust = st.columns(2, gap="large")
        photo = upload.file_id  # widgets reset for every new photo
        addons = table["addons"]
        with adjust:
            dish = st.selectbox("Dish", dishes, format_func=pretty_label, key=f"dish-{photo}",
                                index=dishes.index(predicted) if predicted in dishes else 0,
                                help="Change this if the guess above is wrong.")
            portion = st.segmented_control("Portion", list(PORTIONS), default="Regular",
                                           key=f"portion-{photo}") or "Regular"
            picked = st.pills("Extras", [*addons, OTHER], selection_mode="multi", key=f"extras-{photo}",
                              format_func=lambda k: "Something else" if k == OTHER else addons[k]["label"])
            counts, extra_kcal = {}, 0
            steppers = st.columns(2) if picked else []
            for i, key in enumerate(picked):
                with steppers[i % 2]:
                    if key == OTHER:
                        extra_kcal = st.number_input("Something else (kcal)", min_value=0, max_value=3000,
                                                     value=100, step=10, key=f"kcal-{photo}")
                    else:
                        counts[key] = st.number_input(f"{addons[key]['label']} ({addons[key]['unit']})",
                                                      min_value=1, max_value=10, value=1, step=1,
                                                      key=f"n-{key}-{photo}")
            factor = PORTIONS[portion]
            total = estimate(dish, factor, counts, extra_kcal, table)
            # on phones the card sits above the controls, so repeat the total here
            st.markdown(f'<div class="fl-total-m">Total <b>{round(total["kcal"], -1):,.0f} kcal</b></div>',
                        unsafe_allow_html=True)
        card.markdown(nutrition_html(dish, factor, total, table), unsafe_allow_html=True)

    with st.expander("Details"):
        labels = {p: load_checkpoint(p)[2] for p in checkpoints}
        st.selectbox("Model", checkpoints, key="model_path", format_func=labels.get)
        st.caption(f"{model_label} · {elapsed_ms:.0f} ms on CPU")
        st.dataframe(
            {"Dish": [pretty_label(class_names[i]) for i, _ in top],
             "Probability": [percent(p) for _, p in top]},
            hide_index=True,
        )
        if len(checkpoints) > 1 and st.toggle("Compare all models"):
            rows = []
            for path in checkpoints:
                [(i, p)], ms = predict(image_bytes, path, k=1)
                names, label = load_checkpoint(path)[1:]
                rows.append({"Model": label, "Prediction": pretty_label(names[i]),
                             "Confidence": percent(p), "Time (ms)": round(ms)})
            st.dataframe(rows, hide_index=True)
        if dishes:
            info = table["dishes"][dish]
            st.caption("Nutrition based on " + "; ".join(info["based_on"]) + f" ({table['source']}).")

st.markdown(f'<div class="fl-foot">Recognises {len(class_names)} dishes from the Food-101 dataset.</div>',
            unsafe_allow_html=True)
with st.popover("See all dishes"):
    items = "".join(f"<div>{html.escape(pretty_label(n))}</div>" for n in class_names)
    st.markdown(f'<div class="fl-list">{items}</div>', unsafe_allow_html=True)
