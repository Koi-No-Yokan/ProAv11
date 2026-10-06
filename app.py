import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# KNN INTERACTIVO: ¿DE QUÉ GRUPO SOY?
# Conversión del HTML original a Streamlit
# ============================================================

st.set_page_config(
    page_title="KNN interactivo: ¿De qué grupo soy?",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Estilos
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Bricolage Grotesque', system-ui, sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .result-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .big-result {
        font-size: 1.45rem;
        font-weight: 750;
    }

    .muted {
        opacity: .7;
    }

    .metric-big {
        font-size: 2.2rem;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .7rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Datos / configuración
# -----------------------------
CFG = {
    "abstract": {
        "xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10,
        "xl": "x", "yl": "y",
        "names": ["Azul", "Naranja"],
        "plural": ["azules", "naranjas"],
        "verdict": ["Azul (Clase A)", "Naranja (Clase B)"],
    },
    "rain": {
        "xmin": 30, "xmax": 100, "ymin": 10, "ymax": 35,
        "xl": "Humedad (%)", "yl": "Temperatura (°C)",
        "names": ["No llovió", "Llovió"],
        "plural": ["días sin lluvia", "días con lluvia"],
        "verdict": ["no va a llover", "va a llover"],
    },
}


def rng(seed):
    return random.Random(seed)


def gauss(r):
    return r.gauss(0, 1)


def clamp(v, a, b):
    return min(b, max(a, v))


def r1(v):
    return round(v, 1)


def gen_abstract(seed):
    r = rng(seed * 7919 + 11)
    pts = []

    for _ in range(22):
        x = r1(clamp(3.4 + gauss(r) * 1.5, 0.4, 9.6))
        y = r1(clamp(6.4 + gauss(r) * 1.5, 0.4, 9.6))
        pts.append({"x": x, "y": y, "c": 0, "manual": False, "noise": False})

    for _ in range(22):
        x = r1(clamp(6.6 + gauss(r) * 1.5, 0.4, 9.6))
        y = r1(clamp(3.6 + gauss(r) * 1.5, 0.4, 9.6))
        pts.append({"x": x, "y": y, "c": 1, "manual": False, "noise": False})

    return pts


def gen_rain(seed):
    r = rng(seed * 13 + 5)
    pts = []

    for i in range(60):
        h = round(38 + r.random() * 60)
        t = r1(clamp(12 + r.random() * 20 - (h - 68) * 0.06, 10.5, 34.5))
        z = (h - 66) / 7 - (t - 22) / 4.5

        # Misma idea probabilística del HTML original.
        p_rain = 1 / (1 + math.exp(-z))
        c = 1 if r.random() < p_rain else 0

        pts.append({
            "x": h, "y": t, "c": c,
            "label": f"Día {i + 1}",
            "manual": False,
            "noise": False,
        })

    return pts


def distance(a, b, mode, normalize):
    if mode == "rain" and normalize:
        sx = 1 / (CFG["rain"]["xmax"] - CFG["rain"]["xmin"])
        sy = 1 / (CFG["rain"]["ymax"] - CFG["rain"]["ymin"])
    else:
        sx = sy = 1

    return math.hypot((a["x"] - b["x"]) * sx, (a["y"] - b["y"]) * sy)


def knn(pts, q, k, mode, normalize, tie_rule="none"):
    if not pts or q is None:
        return None

    arr = [
        {
            "p": p,
            "d": distance(p, q, mode, normalize),
        }
        for p in pts
    ]
    arr.sort(key=lambda z: z["d"])
    nb = arr[: min(k, len(arr))]

    votes = [0, 0]
    for n in nb:
        votes[n["p"]["c"]] += 1

    pred = None
    tie = False
    how = ""

    if votes[0] != votes[1]:
        pred = 1 if votes[1] > votes[0] else 0
    else:
        tie = True

        if tie_rule == "nearest":
            pred = nb[0]["p"]["c"]
            how = "nearest"

        elif tie_rule == "weighted":
            weights = [0.0, 0.0]
            for n in nb:
                weights[n["p"]["c"]] += 1 / (n["d"] + 1e-6)

            if weights[0] != weights[1]:
                pred = 1 if weights[1] > weights[0] else 0
                how = "weighted"

    return {
        "nb": nb,
        "votes": votes,
        "pred": pred,
        "tie": tie,
        "how": how,
    }


def current_data():
    return st.session_state.abstract_pts if st.session_state.mode == "abstract" else st.session_state.rain_pts


def current_query():
    return st.session_state.abstract_q if st.session_state.mode == "abstract" else st.session_state.rain_q


def reset_abstract():
    st.session_state.abstract_pts = gen_abstract(st.session_state.seed)
    st.session_state.abstract_q = None
    st.session_state.selected_index = None


def set_query(x, y):
    c = CFG[st.session_state.mode]
    x = clamp(x, c["xmin"], c["xmax"])
    y = clamp(y, c["ymin"], c["ymax"])

    if st.session_state.mode == "rain":
        st.session_state.rain_q = {"x": round(x), "y": r1(y)}
    else:
        st.session_state.abstract_q = {"x": r1(x), "y": r1(y)}


def add_noise():
    pts = st.session_state.abstract_pts
    q = st.session_state.abstract_q

    if q is None:
        q = {"x": 6.7, "y": 3.6}
        st.session_state.abstract_q = q

    base = knn(pts, q, 9, "abstract", False)
    cls = base["pred"]
    opposite = 1 - cls

    nearest = knn(pts, q, 1, "abstract", False)
    dn = nearest["nb"][0]["d"]

    dd = max(0.1, min(0.5, dn * 0.6))
    px, py = q["x"], q["y"]

    for _ in range(30):
        ang = random.random() * math.pi * 2
        px = r1(clamp(q["x"] + math.cos(ang) * dd, 0.2, 9.8))
        py = r1(clamp(q["y"] + math.sin(ang) * dd, 0.2, 9.8))

        if math.hypot(px - q["x"], py - q["y"]) < dn:
            break

    pts.append({
        "x": px, "y": py, "c": opposite,
        "noise": True, "manual": True,
    })

    st.session_state.k = 1
    st.session_state.message = (
        f"Agregaste un dato {'naranja' if opposite else 'azul'} dentro de la nube "
        f"{'naranja' if cls else 'azul'}. Con K=1 el punto cree al dato erróneo. "
        "Sube K a 5 y mira cómo se corrige."
    )


def make_tie():
    pts = st.session_state.abstract_pts
    candidates = []

    x = 0.5
    while x <= 9.5:
        y = 0.5
        while y <= 9.5:
            result = knn(pts, {"x": x, "y": y}, 4, "abstract", False)

            if (
                result["tie"]
                and len(result["nb"]) >= 2
                and result["nb"][0]["d"] < result["nb"][1]["d"] - 1e-9
            ):
                candidates.append({"x": x, "y": y})

            y += 0.5
        x += 0.5

    if not candidates:
        st.session_state.message = (
            'No encontré un empate con este conjunto. Prueba con "Nuevo conjunto".'
        )
        return

    c = random.choice(candidates)
    st.session_state.abstract_q = {"x": r1(c["x"]), "y": r1(c["y"])}
    st.session_state.selected_index = None
    st.session_state.k = 4
    st.session_state.tie_rule = "none"
    st.session_state.message = (
        "Escenario de empate: con K=4, dos vecinos son azules y dos naranjas."
    )


# -----------------------------
# Estado de Streamlit
# -----------------------------
defaults = {
    "mode": "abstract",
    "k": 5,
    "seed": 7,
    "tie_rule": "none",
    "show_dist": False,
    "show_bound": False,
    "normalize": False,
    "place": "query",
    "message": "",
    "selected_index": None,
    "game_on": False,
    "d1": False,
    "b1": False,
    "d2": False,
    "d3": False,
    "score": 0,
    "rain_q": {"x": 68, "y": 23.0},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "abstract_pts" not in st.session_state:
    st.session_state.abstract_pts = gen_abstract(st.session_state.seed)

if "rain_pts" not in st.session_state:
    st.session_state.rain_pts = gen_rain(st.session_state.seed)

if "abstract_q" not in st.session_state:
    st.session_state.abstract_q = None


# -----------------------------
# Acciones de botones
# -----------------------------
def do_mode(mode):
    st.session_state.mode = mode
    st.session_state.selected_index = None
    st.session_state.message = ""
    st.rerun()


def clear_added():
    st.session_state.abstract_pts = [
        p for p in st.session_state.abstract_pts
        if not p.get("manual") and not p.get("noise")
    ]
    st.session_state.message = "Quitaste los datos agregados."
    st.rerun()


def reseed():
    st.session_state.seed += 1
    st.session_state.abstract_pts = gen_abstract(st.session_state.seed)
    st.session_state.abstract_q = None
    st.session_state.selected_index = None
    st.session_state.message = "Nuevo conjunto de datos."
    st.rerun()


# ============================================================
# Encabezado
# ============================================================
if st.session_state.mode == "abstract":
    st.title("¿De qué grupo soy?")
    st.write(
        "Haz clic en el plano para colocar un punto nuevo ❓. "
        "KNN busca sus K vecinos más cercanos y decide por votación."
    )
else:
    st.title("¿Va a llover?")
    st.write(
        "Describe el clima de hoy con dos controles. "
        "KNN busca los días históricos más parecidos y vota."
    )

c1, c2 = st.columns(2)
with c1:
    if st.button(
        "Plano abstracto",
        type="primary" if st.session_state.mode == "abstract" else "secondary",
        use_container_width=True,
    ):
        do_mode("abstract")

with c2:
    if st.button(
        "Caso real: lluvia",
        type="primary" if st.session_state.mode == "rain" else "secondary",
        use_container_width=True,
    ):
        do_mode("rain")


# ============================================================
# Sidebar / controles
# ============================================================
with st.sidebar:
    st.header("Controles")

    st.session_state.k = st.slider(
        "Vecinos que votan (K)",
        min_value=1,
        max_value=15,
        value=st.session_state.k,
        step=1,
        help="Con K impar no hay empates.",
    )

    if st.session_state.mode == "abstract":
        st.radio(
            "Al hacer clic en el plano coloco",
            ["❓ Punto nuevo", "🔵 Dato azul", "🟠 Dato naranja"],
            key="place_ui",
        )

        place_map = {
            "❓ Punto nuevo": "query",
            "🔵 Dato azul": "0",
            "🟠 Dato naranja": "1",
        }
        st.session_state.place = place_map[st.session_state.place_ui]

    else:
        st.session_state.rain_q["x"] = st.slider(
            "Humedad de hoy",
            30, 100,
            int(st.session_state.rain_q["x"]),
            1,
        )

        st.session_state.rain_q["y"] = st.slider(
            "Temperatura de hoy",
            10.0, 35.0,
            float(st.session_state.rain_q["y"]),
            0.1,
        )

    st.session_state.show_dist = st.checkbox(
        "Mostrar distancias",
        value=st.session_state.show_dist,
    )

    st.session_state.show_bound = st.checkbox(
        "Mostrar cómo piensa KNN",
        value=st.session_state.show_bound,
    )

    if st.session_state.mode == "rain":
        st.session_state.normalize = st.checkbox(
            "Normalizar variables",
            value=st.session_state.normalize,
        )

        if st.session_state.normalize:
            st.caption(
                "Cada variable se divide por su rango: humedad y temperatura "
                "pesan igual en la distancia."
            )
        else:
            st.caption(
                "Sin normalizar, la humedad (rango 70) pesa más que la "
                "temperatura (rango 25) en la distancia."
            )

    if st.session_state.mode == "abstract":
        st.divider()

        if st.button("Agregar dato erróneo", use_container_width=True):
            add_noise()
            st.rerun()

        if st.button("Crear empate", use_container_width=True):
            make_tie()
            st.rerun()

        if st.button("Quitar datos agregados", use_container_width=True):
            clear_added()

        if st.button("Nuevo conjunto", use_container_width=True):
            reseed()


# ============================================================
# Cálculo principal
# ============================================================
pts = current_data()
q = current_query()

result = knn(
    pts,
    q,
    st.session_state.k,
    st.session_state.mode,
    st.session_state.normalize,
    st.session_state.tie_rule,
) if q is not None else None


# ============================================================
# Gráfico Plotly
# ============================================================
def make_chart():
    c = CFG[st.session_state.mode]

    fig = go.Figure()

    # Fronteras de decisión aproximadas.
    if st.session_state.show_bound:
        grid = 28
        xs = np.linspace(c["xmin"], c["xmax"], grid)
        ys = np.linspace(c["ymin"], c["ymax"], grid)

        z = []
        for y in ys:
            row = []
            for x in xs:
                rr = knn(
                    pts,
                    {"x": float(x), "y": float(y)},
                    st.session_state.k,
                    st.session_state.mode,
                    st.session_state.normalize,
                    st.session_state.tie_rule,
                )
                row.append(-1 if rr["pred"] is None else rr["pred"])
            z.append(row)

        # Dos mapas transparentes superpuestos para aproximar la zona de decisión.
        fig.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                showscale=False,
                hoverinfo="skip",
                opacity=0.16,
                colorscale=[
                    [0.0, "#2F6FEB"],
                    [0.499, "#2F6FEB"],
                    [0.5, "#EE8A1E"],
                    [1.0, "#EE8A1E"],
                ],
            )
        )

    # Puntos por clase.
    for cls, name, color in [
        (0, CFG[st.session_state.mode]["names"][0], "#2F6FEB"),
        (1, CFG[st.session_state.mode]["names"][1], "#EE8A1E"),
    ]:
        group = [p for p in pts if p["c"] == cls]

        if group:
            fig.add_trace(
                go.Scatter(
                    x=[p["x"] for p in group],
                    y=[p["y"] for p in group],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=9 if st.session_state.mode == "rain" else 11,
                        color=color,
                        line=dict(width=1.5, color="white"),
                    ),
                    customdata=[
                        [
                            p.get("label", ""),
                            "Dato erróneo" if p.get("noise") else (
                                "Dato agregado" if p.get("manual") else ""
                            ),
                        ]
                        for p in group
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "x=%{x}<br>y=%{y}<br>"
                        "%{customdata[1]}<extra></extra>"
                    ),
                )
            )

    # Vecinos K.
    if result:
        nb_points = [n["p"] for n in result["nb"]]

        for cls, color in [(0, "#2F6FEB"), (1, "#EE8A1E")]:
            group = [p for p in nb_points if p["c"] == cls]
            if group:
                fig.add_trace(
                    go.Scatter(
                        x=[p["x"] for p in group],
                        y=[p["y"] for p in group],
                        mode="markers",
                        name=f"Vecinos {CFG[st.session_state.mode]['names'][cls]}",
                        marker=dict(
                            size=17,
                            color="rgba(0,0,0,0)",
                            line=dict(width=2.5, color=color),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Líneas hacia los vecinos.
        if st.session_state.show_dist:
            for n in result["nb"]:
                p = n["p"]
                color = "#EE8A1E" if p["c"] else "#2F6FEB"
                fig.add_trace(
                    go.Scatter(
                        x=[q["x"], p["x"]],
                        y=[q["y"], p["y"]],
                        mode="lines",
                        line=dict(color=color, width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Punto consulta.
    if q:
        pred = result["pred"] if result else None
        qcolor = "#EE8A1E" if pred == 1 else "#2F6FEB" if pred == 0 else "#141C2B"

        fig.add_trace(
            go.Scatter(
                x=[q["x"]],
                y=[q["y"]],
                mode="markers+text",
                text=["?"],
                textposition="middle center",
                name="Punto nuevo" if st.session_state.mode == "abstract" else "Hoy",
                marker=dict(
                    size=25,
                    color="white",
                    line=dict(width=4, color=qcolor),
                ),
                textfont=dict(size=14, color="#141C2B"),
                hovertemplate=(
                    f"{CFG[st.session_state.mode]['xl']}: %{{x}}<br>"
                    f"{CFG[st.session_state.mode]['yl']}: %{{y}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=600,
        margin=dict(l=50, r=15, t=20, b=55),
        xaxis=dict(
            title=c["xl"],
            range=[c["xmin"], c["xmax"]],
            showgrid=True,
            gridcolor="#E9EEF5",
        ),
        yaxis=dict(
            title=c["yl"],
            range=[c["ymin"], c["ymax"]],
            showgrid=True,
            gridcolor="#E9EEF5",
        ),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="closest",
    )

    return fig


left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.plotly_chart(make_chart(), use_container_width=True, config={"displayModeBar": False})

    if st.session_state.message:
        st.info(st.session_state.message)

    st.caption(
        "💡 En el plano abstracto puedes cambiar el tipo de dato colocado "
        "desde los controles. En el caso de lluvia, usa los controles de humedad "
        "y temperatura para mover el punto de hoy."
    )


with right:
    st.subheader("Votación")

    if result is None:
        st.info(
            "Coloca un punto en el plano para ver cómo votan sus vecinos."
        )
    else:
        v0, v1 = result["votes"]
        k_actual = len(result["nb"])

        if result["pred"] is None:
            st.error(
                f"**Empate:** {v0} contra {v1}. El algoritmo no puede decidir."
            )
        else:
            verdict = CFG[st.session_state.mode]["verdict"][result["pred"]]
            extra = ""

            if result["tie"]:
                if result["how"] == "nearest":
                    extra = " — empate resuelto por el vecino más cercano"
                elif result["how"] == "weighted":
                    extra = " — empate resuelto ponderando por distancia"

            st.markdown(
                f'<div class="big-result">{"🔵" if result["pred"] == 0 else "🟠"} '
                f'{("Predicción: " if st.session_state.mode == "rain" else "Clasificado como ")}'
                f'{verdict}{extra}</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.mode == "rain":
            probability = round(v1 / k_actual * 100)
            st.metric(
                "Probabilidad aproximada de lluvia",
                f"{probability}%",
                f"{v1} de {k_actual} días parecidos tuvieron lluvia",
            )

        st.progress(
            v0 / max(1, k_actual),
            text=f"Azul / No llovió: {v0} de {k_actual}",
        )
        st.progress(
            v1 / max(1, k_actual),
            text=f"Naranja / Llovió: {v1} de {k_actual}",
        )

        # Empate
        if result["tie"] and result["pred"] is None:
            st.warning("¿Qué debería hacer el algoritmo?")

            e1, e2, e3 = st.columns(3)

            with e1:
                if st.button("Elegir K impar", use_container_width=True):
                    st.session_state.k = (
                        st.session_state.k - 1
                        if st.session_state.k >= 15
                        else st.session_state.k + 1
                    )
                    st.session_state.message = (
                        f"Con K={st.session_state.k} el número de votos ya "
                        "no puede repartirse en partes iguales."
                    )
                    st.rerun()

            with e2:
                if st.button("Vecino más cercano", use_container_width=True):
                    st.session_state.tie_rule = "nearest"
                    st.session_state.message = (
                        "Desempate: gana la clase del vecino más cercano."
                    )
                    st.rerun()

            with e3:
                if st.button("Ponderar por distancia", use_container_width=True):
                    st.session_state.tie_rule = "weighted"
                    st.session_state.message = (
                        "Desempate: cada voto pesa 1/d, así los vecinos cercanos cuentan más."
                    )
                    st.rerun()

        elif result["tie"] and st.session_state.tie_rule != "none":
            if st.button("Quitar regla de desempate"):
                st.session_state.tie_rule = "none"
                st.rerun()

        # Vecinos
        title = (
            f"Los {k_actual} días más parecidos"
            if st.session_state.mode == "rain"
            else f"Los {k_actual} vecinos más cercanos"
        )
        st.subheader(title)

        for i, nn in enumerate(result["nb"]):
            p = nn["p"]

            if st.session_state.mode == "rain":
                desc = f"{p['label']}: {p['x']}% humedad, {p['y']:.1f} °C"
                cls = CFG["rain"]["names"][p["c"]]
            else:
                desc = f"({p['x']:.1f}, {p['y']:.1f})"
                cls = CFG["abstract"]["names"][p["c"]]

            if st.session_state.show_dist:
                st.markdown(
                    (
                    f"**{'🔵' if p['c'] == 0 else '🟠'} {desc}** — "
                    f"{cls} — d={nn['d']:.3f}"
                    if st.session_state.mode == "rain" and st.session_state.normalize
                    else
                    f"**{'🔵' if p['c'] == 0 else '🟠'} {desc}** — {cls} — d={nn['d']:.2f}"
                )
                )
            else:
                st.write(f"{'🔵' if p['c'] == 0 else '🟠'} {desc} — {cls}")

            if p.get("noise"):
                st.caption("⚠️ Dato erróneo")

        # Comparación con otros K
        st.subheader("¿Y con otro K?")

        ks = [1, 3, 5, 7, 9, 11, 13, 15]
        cols = st.columns(4)

        predictions = []
        for i, kk in enumerate(ks):
            rr = knn(
                pts,
                q,
                kk,
                st.session_state.mode,
                st.session_state.normalize,
                "none",
            )
            predictions.append(rr["pred"])

            with cols[i % 4]:
                if st.button(
                    f"K={kk} → "
                    f"{'🔵' if rr['pred'] == 0 else '🟠' if rr['pred'] == 1 else '⚖️'}",
                    key=f"k_{kk}",
                    use_container_width=True,
                ):
                    st.session_state.k = kk
                    st.session_state.tie_rule = "none"
                    st.rerun()

        if any(p != predictions[0] for p in predictions):
            st.caption(
                "La respuesta cambia según K: un K pequeño es más sensible al "
                "ruido, uno grande suaviza."
            )
        else:
            st.caption(
                "Todos los K coinciden aquí: el punto está bien dentro de un grupo."
            )

        if st.session_state.mode == "rain":
            st.caption("KNN no aprende una fórmula: busca ejemplos parecidos y vota.")


# ============================================================
# Cálculo de distancia
# ============================================================
if result and st.session_state.show_dist:
    st.divider()
    st.subheader("Cálculo de distancia")

    selected = result["nb"][0]
    if st.session_state.selected_index is not None:
        idx = st.session_state.selected_index
        if 0 <= idx < len(result["nb"]):
            selected = result["nb"][idx]

    p = selected["p"]

    if st.session_state.mode == "rain" and st.session_state.normalize:
        dx = (q["x"] - p["x"]) / 70
        dy = (q["y"] - p["y"]) / 25
        d = math.sqrt(dx * dx + dy * dy)

        st.latex(
            rf"d = \sqrt{{\left(\frac{{{q['x']} - {p['x']}}}{{70}}\right)^2 "
            rf"+ \left(\frac{{{q['y']:.1f} - {p['y']:.1f}}}{{25}}\right)^2}}"
        )
        st.write(f"Distancia = **{d:.3f}**")
    else:
        dx = q["x"] - p["x"]
        dy = q["y"] - p["y"]
        d = math.sqrt(dx * dx + dy * dy)

        st.latex(
            rf"d = \sqrt{{({q['x']:.1f} - {p['x']:.1f})^2 + "
            rf"({q['y']:.1f} - {p['y']:.1f})^2}}"
        )
        st.write(f"Distancia = **{d:.2f}**")


# ============================================================
# Modo juego
# ============================================================
if st.session_state.mode == "abstract":
    st.divider()
    st.subheader("Modo juego")

    st.session_state.game_on = st.checkbox(
        "Activar retos",
        value=st.session_state.game_on,
    )

    if st.session_state.game_on:
        # Comprobación de retos.
        if q is not None:
            r3 = knn(pts, q, 3, "abstract", False)
            if not st.session_state.d1 and r3["pred"] == 0:
                st.session_state.d1 = True
                st.session_state.score += 10

            if (
                not st.session_state.b1
                and r3["pred"] == 0
                and r3["votes"] == [2, 1]
            ):
                st.session_state.b1 = True
                st.session_state.score += 5

            r1n = knn(pts, q, 1, "abstract", False)
            r7 = knn(pts, q, 7, "abstract", False)

            if (
                not st.session_state.d2
                and r1n["pred"] is not None
                and r7["pred"] is not None
                and r1n["pred"] != r7["pred"]
            ):
                st.session_state.d2 = True
                st.session_state.score += 15

            near = r1n["nb"][0]["p"]
            if (
                not st.session_state.d3
                and near.get("manual")
                and near["c"] != r7["pred"]
            ):
                st.session_state.d3 = True
                st.session_state.score += 15

        st.write(f"**Puntos: {st.session_state.score} de 45**")

        st.markdown(
            f"- {'✅' if st.session_state.d1 else '⬜'} "
            "**Reto 1:** coloca un punto que sea Azul con K=3. "
            "(10 puntos + bonus de 5 si la votación es 2 a 1.)"
        )
        st.markdown(
            f"- {'✅' if st.session_state.d2 else '⬜'} "
            "**Reto 2:** encuentra un lugar donde K=1 y K=7 den respuestas distintas. "
            "(15 puntos)"
        )
        st.markdown(
            f"- {'✅' if st.session_state.d3 else '⬜'} "
            "**Reto 3:** con ❓ colocado, agrega un dato azul o naranja que engañe a K=1. "
            "(15 puntos)"
        )

        if st.button("Reiniciar retos"):
            st.session_state.d1 = False
            st.session_state.b1 = False
            st.session_state.d2 = False
            st.session_state.d3 = False
            st.session_state.score = 0
            st.session_state.abstract_q = None
            st.session_state.abstract_pts = [
                p for p in st.session_state.abstract_pts
                if not p.get("manual") and not p.get("noise")
            ]
            st.session_state.place = "query"
            st.rerun()


# ============================================================
# Guía
# ============================================================
with st.expander("Guía para la clase"):
    if st.session_state.mode == "abstract":
        st.markdown(
            """
            1. Haz clic en el plano para colocar ❓ y arrástralo hacia un grupo.
            2. Cambia K de 1 a 9 y mira cuándo cambia la clasificación.
            3. Pulsa **Agregar dato erróneo**: K=1 se equivoca y K=5 lo corrige.
            4. Activa **Mostrar distancias** para ver los cálculos.
            5. Pulsa **Crear empate** y decide cómo resolverlo.
            6. Activa **Mostrar cómo piensa KNN** y compara K=1 con K=15.
            """
        )
    else:
        st.markdown(
            """
            1. Mueve humedad y temperatura para describir el día de hoy.
            2. Revisa los días más parecidos y cuántos tuvieron lluvia.
            3. Cambia K y observa cómo cambia la probabilidad.
            4. Activa y desactiva **Normalizar variables**. ¿Cambian los vecinos? ¿Por qué?
            """
        )
