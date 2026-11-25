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
from foodlens.nutrition import DAILY_KCAL, PORTIONS, daily_value, estimate, load_table

CHECKPOINT_DIR = Path(os.environ.get("FOODLENS_CHECKPOINTS", Path(__file__).parent / "checkpoints"))
DEFAULT_CHECKPOINT = "best.pt"
SURE, LIKELY = 0.70, 0.40  # confidence bands shown to the user
EVAL_TF = build_transforms(train=False)
OTHER = "other"
# rows of the nutrition panel: (key, label, unit, indented)
PANEL_ROWS = [("fat_g", "Total Fat", "g", False), ("sodium_mg", "Sodium", "mg", False),
              ("carbs_g", "Total Carbohydrate", "g", False), ("fiber_g", "Dietary Fiber", "g", True),
              ("sugar_g", "Total Sugars", "g", True), ("protein_g", "Protein", "g", False)]

st.set_page_config(page_title="FoodLens", page_icon=":material/restaurant:", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3rem; max-width: 820px;}
    .fl-title {font-weight: 900; font-size: 3.1rem; letter-spacing: -0.04em; line-height: .95; color: #111;}
    .fl-sub {color: #444; font-size: 1.05rem; margin-top: .55rem; padding-bottom: 1.1rem;
             border-bottom: 10px solid #111; margin-bottom: 1.4rem;}
    .fl-kicker {font-size: .75rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; color: #555;}
    .fl-result {margin-bottom: 1.25rem;}
    .fl-dish {font-weight: 900; font-size: 2.6rem; letter-spacing: -0.035em; line-height: 1; color: #111;
              margin: .25rem 0 .55rem;}
    .fl-note {color: #333; margin-bottom: .9rem;}
    .fl-row {display: flex; justify-content: space-between; gap: 1rem; font-size: .95rem; font-weight: 600;
             color: #111; margin-top: .55rem; padding-top: .45rem; border-top: 1px solid #111;}
    .fl-row span:last-child {font-variant-numeric: tabular-nums;}
    .fl-bar {height: 10px; background: #ECECE8; margin-top: .35rem;}
    .fl-bar i {display: block; height: 100%; background: #D62718;}
    .fl-bar.alt i {background: #A3A39C;}
    .fl-gap {height: .6rem; border-top: 1px solid #111; margin-top: 1.6rem;}
    .nf {border: 1px solid #111; padding: .3rem .6rem .55rem; color: #111; background: #fff; margin-bottom: 1.2rem;}
    .nf-h {font-weight: 900; font-size: 2.2rem; letter-spacing: -0.03em; line-height: 1.05;}
    .nf-dish {font-size: .95rem; padding: .15rem 0 .1rem;}
    .nf-serv {display: flex; justify-content: space-between; gap: .75rem; font-weight: 800; font-size: 1rem;}
    .nf-serv span:last-child {text-align: right;}
    .nf-incl {font-size: .82rem; color: #333; padding-top: .15rem;}
    .nf-thick {height: 10px; background: #111; margin: .35rem 0 .25rem;}
    .nf-med {height: 5px; background: #111; margin: .3rem 0 .2rem;}
    .nf-amt {font-size: .78rem; font-weight: 800;}
    .nf-cal {display: flex; justify-content: space-between; align-items: baseline; font-weight: 900;}
    .nf-cal span {font-size: 1.75rem; letter-spacing: -0.02em;}
    .nf-cal b {font-size: 2.7rem; letter-spacing: -0.03em; font-variant-numeric: tabular-nums;}
    .nf-dvh {text-align: right; font-size: .78rem; font-weight: 800; padding-bottom: .2rem;
             border-bottom: 1px solid #111;}
    .nf-row {display: flex; justify-content: space-between; font-size: .93rem; padding: .22rem 0;
             border-bottom: 1px solid #111; font-variant-numeric: tabular-nums;}
    .nf-row.sub {padding-left: 1.1rem;}
    .nf-row b {font-weight: 800;}
    .nf-row .dv {font-weight: 800;}
    .nf-foot {font-size: .74rem; line-height: 1.4; margin-top: .45rem;}
    .fl-total-m {display: none;}
    .fl-foot {color: #555; font-size: .85rem; margin-top: 2.5rem; padding-top: .6rem; border-top: 1px solid #111;}
    .fl-list {columns: 3 10rem; font-size: .85rem; color: #111; line-height: 1.6;}
    [data-testid="stImage"] img {border: 1px solid #111; max-height: 420px; object-fit: cover;}
    [data-testid="stButtonGroup"] > div {flex-wrap: wrap; overflow-x: visible; row-gap: .45rem;}
    @media (max-width: 640px) {
      .block-container {padding-top: 2rem;}
      .fl-title {font-size: 2.5rem;}
      .fl-dish {font-size: 2.2rem;}
      [data-testid="stImage"] img {max-height: 240px;}
      .fl-total-m {display: flex; justify-content: space-between; align-items: baseline; font-weight: 900;
                   border-top: 10px solid #111; padding-top: .35rem; margin-top: .3rem;}
      .fl-total-m span {font-size: 1.3rem;} .fl-total-m b {font-size: 1.8rem; letter-spacing: -0.03em;}
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


def nutrition_html(dish: str, factor: float, total: dict, table: dict, added: list[str]) -> str:
    info = table["dishes"][dish]
    serving = info["serving"] + (f" × {factor:g}" if factor != 1 else "")
    grams = f"{info['grams'] * factor:,.0f} g"
    included = f'<div class="nf-incl">Includes {html.escape(", ".join(added))}</div>' if added else ""
    rows = []
    for key, label, unit, sub in PANEL_ROWS:
        amount = f"{round(total[key], -1):,.0f}{unit}" if unit == "mg" else f"{total[key]:.0f}{unit}"
        dv = daily_value(total, key) if key != "protein_g" else None
        name = html.escape(label) if sub else f"<b>{html.escape(label)}</b>"
        rows.append(f'<div class="nf-row{" sub" if sub else ""}"><span>{name} {amount}</span>'
                    f'<span class="dv">{f"{dv:.0%}" if dv is not None else ""}</span></div>')
    return (f'<div class="nf"><div class="nf-h">Nutrition Facts</div>'
            f'<div class="nf-dish">{html.escape(pretty_label(dish))}, estimated</div>'
            f'<div class="nf-serv"><span>Serving size</span><span>{html.escape(serving)} ({grams})</span></div>'
            f'{included}<div class="nf-thick"></div><div class="nf-amt">Amount per serving</div>'
            f'<div class="nf-cal"><span>Calories</span><b>{round(total["kcal"], -1):,.0f}</b></div>'
            f'<div class="nf-med"></div><div class="nf-dvh">% Daily Value*</div>{"".join(rows)}'
            f'<div class="nf-thick"></div>'
            f'<div class="nf-foot">* Based on a {DAILY_KCAL:,}-calorie day. Estimated from USDA food data '
            f'for a typical serving; real meals vary.</div></div>')


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
    left.image(image, width="stretch")
    right.markdown(result_html(top, class_names), unsafe_allow_html=True)

    predicted = class_names[top[0][0]]
    dishes = sorted((d for d in class_names if d in table["dishes"]), key=pretty_label)
    if dishes:
        st.markdown('<div class="fl-gap"></div>', unsafe_allow_html=True)
        panel, adjust = st.columns(2, gap="large")
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
            drinks = table.get("drinks", {})
            drink = st.selectbox("Drink", [None, *drinks], key=f"drink-{photo}",
                                 format_func=lambda k: "No drink" if k is None
                                 else f"{drinks[k]['label']} · {drinks[k]['serving']}")
            factor = PORTIONS[portion]
            total = estimate(dish, portion=factor, extras=counts, extra_kcal=extra_kcal, drink=drink, table=table)
            # on phones the panel sits above the controls, so repeat the total here
            st.markdown(f'<div class="fl-total-m"><span>Calories</span>'
                        f'<b>{round(total["kcal"], -1):,.0f}</b></div>', unsafe_allow_html=True)
        added = [f"{addons[k]['label'].lower()} ×{n}" for k, n in counts.items()]
        if extra_kcal:
            added.append(f"{extra_kcal} kcal more")
        if drink:
            added.append(f"{drinks[drink]['label'].lower()} — {drinks[drink]['serving']}")
        panel.markdown(nutrition_html(dish, factor, total, table, added), unsafe_allow_html=True)

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
            sources = list(table["dishes"][dish]["based_on"])
            if drink:
                sources += table["drinks"][drink]["based_on"]
            st.caption("Nutrition based on " + "; ".join(sources) + f" ({table['source']}).")

st.markdown(f'<div class="fl-foot">Recognises {len(class_names)} dishes from the Food-101 dataset.</div>',
            unsafe_allow_html=True)
with st.popover("See all dishes"):
    items = "".join(f"<div>{html.escape(pretty_label(n))}</div>" for n in class_names)
    st.markdown(f'<div class="fl-list">{items}</div>', unsafe_allow_html=True)
