import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, UnidentifiedImageError
import requests


# =========================
# CONFIGURACIÓN DE LOGOS
# =========================
HEADER_LOGO_2_URL = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%20reach.png?alt=media&token=a3f686ba-dd6e-470c-ae73-3c950d4b1b52"
HEADER_LOGO_URL = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%205reach%20logo.png?alt=media&token=303012df-f5f0-4852-a6ce-19017bc4b040"
FOOTER_LOGO_1_URL = "https://effective.com.mx/wp-content/uploads/2024/10/logo-main-nav.png"
FOOTER_LOGO_2_URL = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Farchimed%20logo.png?alt=media&token=36551e78-097d-4372-ba1f-8520ec375b8c"


# =========================
# FUNCIONES AUXILIARES
# =========================
def render_small_fig(fig):
    """Renderiza una figura matplotlib como imagen PNG en memoria."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


def load_logo(url, height_target):
    """
    Carga un logo desde URL de forma segura.
    Si la URL no devuelve una imagen válida, regresa None para evitar que truene la app.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if "image" not in content_type:
            return None

        logo = Image.open(BytesIO(response.content)).convert("RGBA")

        ratio = height_target / logo.height
        new_size = (int(logo.width * ratio), height_target)

        return logo.resize(new_size, Image.LANCZOS)

    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError):
        return None


def add_header_footer(image_bytes):
    """
    Agrega header y footer a una imagen PNG.
    Versión corregida: si algún logo falla, omite ese logo y continúa.
    """
    image_bytes.seek(0)
    original_img = Image.open(image_bytes).convert("RGB")
    width, height = original_img.size

    header_height = 60
    footer_height = 60

    header_logo = load_logo(HEADER_LOGO_URL, 60)
    header_logo2 = load_logo(HEADER_LOGO_2_URL, 45)
    footer_logo1 = load_logo(FOOTER_LOGO_1_URL, footer_height - 15)
    footer_logo2 = load_logo(FOOTER_LOGO_2_URL, footer_height - 15)

    new_img = Image.new(
        "RGB",
        (width, height + header_height + footer_height),
        "white"
    )
    draw = ImageDraw.Draw(new_img)

    # Header
    draw.rectangle([0, 0, width, header_height], fill=(0, 0, 0))

    x_header = 10
    if header_logo is not None:
        y_header = int((header_height - header_logo.height) / 2)
        new_img.paste(header_logo, (x_header, y_header), header_logo)
        x_header += header_logo.width + 10

    if header_logo2 is not None:
        y_header2 = int((header_height - header_logo2.height) / 2)
        new_img.paste(header_logo2, (x_header, y_header2), header_logo2)

    # Imagen principal
    new_img.paste(original_img, (0, header_height))

    # Footer
    footer_y = height + header_height
    draw.rectangle([0, footer_y, width, footer_y + footer_height], fill=(0, 0, 0))

    text = "© 2025 Effective  |  Designed by ArchimED"
    draw.text((15, footer_y + 20), text, fill="white")

    spacing = 10
    x_right = width - 10

    if footer_logo2 is not None:
        x_right -= footer_logo2.width
        y_footer2 = footer_y + int((footer_height - footer_logo2.height) / 2)
        new_img.paste(footer_logo2, (x_right, y_footer2), footer_logo2)
        x_right -= spacing

    if footer_logo1 is not None:
        x_right -= footer_logo1.width
        y_footer1 = footer_y + int((footer_height - footer_logo1.height) / 2)
        new_img.paste(footer_logo1, (x_right, y_footer1), footer_logo1)

    output = BytesIO()
    new_img.save(output, format="PNG")
    output.seek(0)
    return output


def build_asymmetric_overlap_matrix(reach_dict, duplication_factor):
    """Construye matriz de solapamiento no simétrica por comparación directa."""
    medios = list(reach_dict.keys())
    n = len(medios)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            reach_i = reach_dict[medios[i]]
            reach_j = reach_dict[medios[j]]

            if i == j:
                matrix[i][j] = reach_i
            else:
                cume = reach_i + reach_j - duplication_factor * (reach_i * reach_j)
                matrix[i][j] = cume - reach_j

    return pd.DataFrame(matrix, index=medios, columns=medios)


# =========================
# CONFIGURACIÓN STREAMLIT
# =========================
st.set_page_config(
    page_title="Effective Reach",
    page_icon=HEADER_LOGO_URL,
    layout="wide"
)


st.markdown(f"""
    <div style="background-color:#000000; padding: 10px 30px; position: fixed; top: 0; left: 0; width: 100%; display: flex; align-items: center; z-index: 999;">
        <div>
            <img src="{HEADER_LOGO_URL}" height="70">
        </div>
        <div>
            <img src="{HEADER_LOGO_2_URL}" height="50">
        </div>
    </div>
    <div style="margin-top:80px;"></div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        body { background-color: #ffffff; }
        h1, h2, h3, h4, h5 { color: #FDB813; }
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# =========================
# DATA INICIAL
# =========================
if "reach_data" not in st.session_state or st.session_state["reach_data"].empty:
    st.session_state["reach_data"] = pd.DataFrame({
        "Medio": [
            "TV ABIERTA", "TV PAGA", "TV LOCAL", "OOH", "IMPRESOS",
            "RADIO", "SOCIAL", "SEARCH", "PROGRAMATIC", "CTV"
        ],
        "Reach_Individual": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    })


col1, col2 = st.columns([1, 3])


# =========================
# PANEL IZQUIERDO
# =========================
with col1:
    st.markdown("### Panel de control")

    if st.button("Restablecer catálogo inicial"):
        st.session_state["reach_data"] = pd.DataFrame({
            "Medio": [
                "TV ABIERTA", "TV PAGA", "TV LOCAL", "OOH", "IMPRESOS",
                "RADIO", "SOCIAL", "SEARCH", "PROGRAMATIC", "CTV"
            ],
            "Reach_Individual": [0.60, 0.40, 0.30, 0.25, 0.15, 0.35, 0.50, 0.30, 0.20, 0.10]
        })
        st.success("Catálogo restablecido con valores predefinidos.")

    st.markdown("---")
    st.markdown("#### Editar medios")

    df_editable = st.session_state["reach_data"][["Medio", "Reach_Individual"]].copy()
    df_editable["Reach (%)"] = df_editable["Reach_Individual"] * 100
    df_editable = df_editable.drop(columns=["Reach_Individual"])

    edited_df = st.data_editor(
        df_editable,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Reach (%)": st.column_config.NumberColumn(
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                step=0.1
            ),
            "Medio": st.column_config.TextColumn()
        }
    )

    errores = []

    if edited_df["Medio"].isnull().any() or edited_df["Medio"].astype(str).str.strip().eq("").any():
        errores.append("⚠️ Hay medios con nombre vacío.")

    if (
        edited_df["Reach (%)"].isnull().any()
        or (edited_df["Reach (%)"] < 0).any()
        or (edited_df["Reach (%)"] > 100).any()
    ):
        errores.append("⚠️ Algunos valores de reach individual son inválidos.")

    if edited_df["Medio"].duplicated().any():
        errores.append("⚠️ Hay medios con nombres duplicados.")

    if errores:
        for err in errores:
            st.error(err)
    else:
        edited_df["Reach_Individual"] = edited_df["Reach (%)"] / 100
        st.session_state["reach_data"] = edited_df[["Medio", "Reach_Individual"]]

    st.markdown("---")
    st.markdown("#### Factor de duplicidad")
    duplicidad = st.slider("", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    st.caption("Este factor ajusta la penalización por superposición entre medios.")


# =========================
# CONTENIDO PRINCIPAL
# =========================
with col2:
    df = st.session_state["reach_data"].copy()

    data_valida = (
        not df.empty
        and not df["Medio"].duplicated().any()
        and not df["Medio"].isnull().any()
        and not df["Medio"].astype(str).str.strip().eq("").any()
    )

    if data_valida:
        # Cálculo reach cume
        reach_cume = []
        for i, r in enumerate(df["Reach_Individual"]):
            if i == 0:
                acumulado = r
            else:
                acumulado = min(
                    reach_cume[-1] + r - duplicidad * (reach_cume[-1] * r),
                    1.0
                )
            reach_cume.append(acumulado)

        df["Reach_Cume"] = reach_cume

        st.markdown("### Tabla de Reach acumulado")
        st.dataframe(
            df.style.format({
                "Reach_Individual": "{:.1%}",
                "Reach_Cume": "{:.1%}"
            }),
            use_container_width=True
        )

        total_final = df["Reach_Cume"].iloc[-1]
        st.markdown(
            f"<h4 style='color:#FDB813;'>Reach Total Acumulado: <b>{total_final:.1%}</b></h4>",
            unsafe_allow_html=True
        )

        # =========================
        # GRÁFICO 1: CURVA
        # =========================
        st.markdown("---")
        st.markdown("### Curva de Reach Cume")

        fig, ax = plt.subplots(figsize=(10, 3))

        x = np.arange(0, len(df) + 1)
        y = [0.0] + df["Reach_Cume"].tolist()

        ax.plot(x, y, marker="o", color="#FDB813", label="Reach Cume")

        for i in range(1, len(x)):
            ax.text(x[i], y[i] + 0.01, f"{y[i]:.2%}", ha="center", fontsize=10)

        ax.set_xticks(np.arange(1, len(df) + 1))
        ax.set_xticklabels(df["Medio"], rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Reach acumulado (%)")
        ax.set_xlabel("Medios")
        ax.set_ylim(0, min(1.05, max(df["Reach_Cume"]) + 0.12))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        curva_buf = render_small_fig(fig)
        st.image(curva_buf)

        curva_download = add_header_footer(curva_buf)
        st.download_button(
            "⬇️ Descargar gráfico Curva como PNG",
            data=curva_download.getvalue(),
            file_name="curva_reach_cume.png",
            mime="image/png"
        )

        plt.close(fig)

        # =========================
        # GRÁFICO 2: BARRAS
        # =========================
        st.markdown("---")
        st.markdown("### Reach Report")

        df_bar = df[["Medio", "Reach_Individual"]].copy()
        df_bar.loc[len(df_bar)] = ["Net campaign reach", total_final]
        df_bar = df_bar[::-1]

        fig_bar, ax_bar = plt.subplots(figsize=(10, 3))

        bar_colors = ["#FDB813" if m == "Net campaign reach" else "black" for m in df_bar["Medio"]]

        ax_bar.barh(df_bar["Medio"], df_bar["Reach_Individual"], color=bar_colors)

        for i, v in enumerate(df_bar["Reach_Individual"]):
            ax_bar.text(v + 0.01, i, f"{v:.2%}", va="center")

        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Reach (%)")
        ax_bar.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        barras_buf = render_small_fig(fig_bar)
        st.image(barras_buf)

        barras_download = add_header_footer(barras_buf)
        st.download_button(
            "⬇️ Descargar gráfico Barras como PNG",
            data=barras_download.getvalue(),
            file_name="barras_reach_individual.png",
            mime="image/png"
        )

        plt.close(fig_bar)

        # =========================
        # GRÁFICO 3: HEATMAP
        # =========================
        st.markdown("---")
        st.markdown("### Heatmap de Solapamiento (No Simétrico por Comparación Directa)")

        reach_dict = dict(zip(df["Medio"], df["Reach_Individual"]))
        overlap_df = build_asymmetric_overlap_matrix(reach_dict, duplicidad)

        percent_df = overlap_df.copy() * 100
        percent_df = percent_df.round(2).T

        custom_cmap = LinearSegmentedColormap.from_list(
            "naranja_dorado",
            ["#FFFFFF", "#FDBA12"]
        )

        fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
        cax = ax_hm.matshow(percent_df.values, cmap=custom_cmap)

        ax_hm.set_xticks(range(len(percent_df.columns)))
        ax_hm.set_xticklabels(percent_df.columns, rotation=45, ha="left")
        ax_hm.set_yticks(range(len(percent_df.index)))
        ax_hm.set_yticklabels(percent_df.index)

        for (i, j), val in np.ndenumerate(percent_df.values):
            if i == j:
                ax_hm.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="black")
                )
                ax_hm.text(
                    j, i, f"{val:.2f}%",
                    ha="center", va="center",
                    color="white", fontweight="bold"
                )
            else:
                ax_hm.text(
                    j, i, f"{val:.2f}%",
                    ha="center", va="center",
                    color="black"
                )

        cbar = fig_hm.colorbar(cax)
        cbar.set_label("Contribución (%)", rotation=270, labelpad=15)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))

        fig_hm.tight_layout()
        heatmap_buf = render_small_fig(fig_hm)

        st.image(heatmap_buf)

        heatmap_download = add_header_footer(heatmap_buf)
        st.download_button(
            "⬇️ Descargar Heatmap Comparación (no simétrico) como PNG",
            data=heatmap_download.getvalue(),
            file_name="heatmap_comparacion_nosimetrico.png",
            mime="image/png"
        )

        plt.close(fig_hm)

        # =========================
        # PDF COMPLETO
        # =========================
        st.markdown("---")
        st.markdown("### Descarga de reporte PDF")

        pdf_buffer = BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            for image_buf in [curva_buf, barras_buf, heatmap_buf]:
                img_buf = add_header_footer(image_buf)
                img_buf.seek(0)

                img = Image.open(img_buf).convert("RGB")

                fig_pdf = plt.figure(figsize=(img.width / 100, img.height / 100))
                ax_pdf = fig_pdf.add_axes([0, 0, 1, 1])
                ax_pdf.axis("off")
                ax_pdf.imshow(img)

                pdf.savefig(fig_pdf)
                plt.close(fig_pdf)

        pdf_buffer.seek(0)

        st.download_button(
            "📄 Descargar reporte completo en PDF",
            data=pdf_buffer.getvalue(),
            file_name="report_reach_cume.pdf",
            mime="application/pdf"
        )

    else:
        st.info(
            "Revisa que todos los medios tengan nombre, valores válidos y no estén duplicados para generar el reporte."
        )


# =========================
# FOOTER FIJO DE LA APP
# =========================
st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)

st.markdown(f"""
    <div style="background-color:#000000; padding: 10px 30px; position: fixed; bottom: 0; left: 0; width: 100%; display: flex; justify-content: space-between; align-items: center; z-index: 100;">
        <div style="color: white; font-size: 14px;">
            © 2025 Effective &nbsp;&nbsp;|&nbsp;&nbsp; Designed by ArchimED
        </div>
        <div style="display: flex; gap: 10px;">
            <img src="{FOOTER_LOGO_1_URL}" height="30">
            <img src="{FOOTER_LOGO_2_URL}" height="40">
        </div>
    </div>
""", unsafe_allow_html=True)
