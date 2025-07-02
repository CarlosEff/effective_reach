import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import streamlit as st
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
import requests
# Función para renderizar figura matplotlib como imagen PNG
def render_small_fig(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# Función para agregar header y footer a imagen PNG
def add_header_footer(image_bytes):
    # URLs de logos

    header_logo2_url = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%20reach.png?alt=media&token=a3f686ba-dd6e-470c-ae73-3c950d4b1b52"
    header_logo_url = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%205reach%20logo.png?alt=media&token=303012df-f5f0-4852-a6ce-19017bc4b040"
    footer_logo1_url = "https://effective.com.mx/wp-content/uploads/2024/10/logo-main-nav.png"
    footer_logo2_url = "https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Farchimed%20logo.png?alt=media&token=36551e78-097d-4372-ba1f-8520ec375b8c"

    original_img = Image.open(image_bytes).convert("RGB")
    width, height = original_img.size

    def load_logo(url, height_target):
        r = requests.get(url)
        logo = Image.open(BytesIO(r.content)).convert("RGBA")
        ratio = height_target / logo.height
        new_size = (int(logo.width * ratio), height_target)
        return logo.resize(new_size, Image.LANCZOS)

    header_height = 60
    footer_height = 60
    header_logo = load_logo(header_logo_url, 60)
    header_logo2 = load_logo(header_logo2_url, 45)
    footer_logo1 = load_logo(footer_logo1_url, footer_height - 15)
    footer_logo2 = load_logo(footer_logo2_url, footer_height - 15)
    new_img = Image.new("RGB", (width, height + header_height + footer_height), "white")
    draw = ImageDraw.Draw(new_img)

    # HEADER
    draw.rectangle([0, 0, width, header_height], fill=(0, 0, 0))
    new_img.paste(header_logo, (10, int((header_height - header_logo.height) / 2)), header_logo)
    new_img.paste(header_logo2, (10 + header_logo.width + 10, int((header_height - header_logo2.height) / 2)),
                  header_logo2)

    # FOOTER
    draw.rectangle([0, height + header_height, width, height + header_height + footer_height], fill=(0, 0, 0))

    # Texto izquierda
    text = "© 2025 Effective  |  Designed by ArchimED"
    draw.text((15, height + header_height + 20), text, fill="white")

    # Logos derecha (con separación)
    spacing = 10
    x2 = width - footer_logo2.width - 10
    x1 = x2 - footer_logo1.width - spacing
    y = height + header_height + int((footer_height - footer_logo1.height) / 2)

    new_img.paste(footer_logo1, (x1, y), footer_logo1)
    new_img.paste(footer_logo2, (x2, y), footer_logo2)

    # Pega el gráfico
    new_img.paste(original_img, (0, header_height))

    output = BytesIO()
    new_img.save(output, format="PNG")
    output.seek(0)
    return output

# Configuración de página
st.set_page_config(page_title="Reach Cume Simulator", layout="wide")

# Header
st.markdown("""
    <div style="background-color:#000000; padding: 10px 30px; position: fixed; top: 0; left: 0; width: 100%; display: flex; align-items: center; z-index: 999;">
        <div>
            <img src="https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%205reach%20logo.png?alt=media&token=303012df-f5f0-4852-a6ce-19017bc4b040" height="70">
        </div>
        <div>
            <img src="https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Feffective%20reach.png?alt=media&token=a3f686ba-dd6e-470c-ae73-3c950d4b1b52" height="50">
        </div>
    </div>
    <div style="margin-top:80px;"></div>
""", unsafe_allow_html=True)

# Estilos
st.markdown("""
    <style>
        body { background-color: #ffffff; }
        h1, h2, h3, h4, h5 { color: #FDB813; }
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Datos base
if "reach_data" not in st.session_state or st.session_state["reach_data"].empty:
    st.session_state["reach_data"] = pd.DataFrame({
        "Medio": ["TV ABIERTA", "TV PAGA", "TV LOCAL", "OOH", "IMPRESOS", "RADIO", "SOCIAL", "SEARCH", "PROGRAMATIC", "CTV"],
        "Reach_Individual": [0.60, 0.40, 0.30, 0.25, 0.15, 0.35, 0.50, 0.30, 0.20, 0.10]
    })

# Layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Panel de control")
    if st.button("Restablecer catálogo inicial"):
        st.session_state["reach_data"] = pd.DataFrame({
            "Medio": ["TV ABIERTA", "TV PAGA", "TV LOCAL", "OOH", "IMPRESOS", "RADIO", "SOCIAL", "SEARCH", "PROGRAMATIC", "CTV"],
            "Reach_Individual": [0.60, 0.40, 0.30, 0.25, 0.15, 0.35, 0.50, 0.30, 0.20, 0.10]
        })
        st.success("Catálogo restablecido con valores predefinidos.")
    st.markdown("---")
    st.markdown("#### Editar medios")
    df_editable = st.session_state["reach_data"][["Medio", "Reach_Individual"]]
    edited_df = st.data_editor(
        df_editable,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Reach_Individual": st.column_config.NumberColumn(format="%.3f", min_value=0.0, max_value=1.0, step=0.001),
            "Medio": st.column_config.TextColumn()
        }
    )

    errores = []
    if edited_df["Medio"].isnull().any() or edited_df["Medio"].str.strip().eq("").any():
        errores.append("⚠️ Hay medios con nombre vacío.")
    if edited_df["Reach_Individual"].isnull().any() or (edited_df["Reach_Individual"] < 0).any() or (edited_df["Reach_Individual"] > 1).any():
        errores.append("⚠️ Algunos valores de reach individual son inválidos.")
    if edited_df["Medio"].duplicated().any():
        errores.append("⚠️ Hay medios con nombres duplicados.")

    if errores:
        for err in errores:
            st.error(err)
    else:
        st.session_state["reach_data"] = edited_df

    st.markdown("---")
    st.markdown("#### Factor de duplicidad")
    duplicidad = st.slider("", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    st.caption("Este factor ajusta la penalización por superposición entre medios.")

with col2:
    df = st.session_state["reach_data"].copy()
    if not df.empty and not df["Medio"].duplicated().any() and not df["Medio"].isnull().any() and not df["Medio"].str.strip().eq("").any():
        reach_cume = []
        for i, r in enumerate(df["Reach_Individual"]):
            acumulado = r if i == 0 else min(reach_cume[-1] + r - duplicidad * (reach_cume[-1] * r), 1.0)
            reach_cume.append(acumulado)
        df["Reach_Cume"] = reach_cume

        st.markdown("### Tabla de Reach acumulado")
        st.dataframe(df.style.format({
            "Reach_Individual": "{:.1%}",
            "Reach_Cume": "{:.1%}"
        }), use_container_width=True)

        total_final = df["Reach_Cume"].iloc[-1]
        st.markdown(f"<h4 style='color:#FDB813;'>Reach Total Acumulado: <b>{total_final:.1%}</b></h4>", unsafe_allow_html=True)

        # Gráfico Curva
        st.markdown("---")
        st.markdown("### Curva de Reach Cume")
        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.arange(1, len(df) + 1)
        ax.plot(x, df["Reach_Cume"], marker='o', color='#FDB813', label="Reach Cume")
        for i, val in enumerate(df["Reach_Cume"]):
            ax.text(x[i], val + 0.01, f"{val:.2%}", ha='center', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Medio"], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Reach acumulado (%)")
        ax.set_xlabel("Medios")
        ax.set_ylim(0, min(1.05, max(df["Reach_Cume"]) + 0.05))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        curva_buf = render_small_fig(fig)
        st.image(curva_buf)
        st.download_button("⬇️ Descargar gráfico Curva como PNG", data=add_header_footer(curva_buf), file_name="curva_reach_cume.png", mime="image/png")

        # Gráfico Barras
        st.markdown("---")
        st.markdown("### Reach Report")
        df_bar = df[["Medio", "Reach_Individual"]].copy()
        df_bar.loc[len(df_bar)] = ["Net campaign reach", total_final]
        df_bar = df_bar[::-1]
        colors = ["#FDB813" if m == "Net campaign reach" else "steelblue" for m in df_bar["Medio"]]
        fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
        ax_bar.barh(df_bar["Medio"], df_bar["Reach_Individual"], color=colors)
        for i, v in enumerate(df_bar["Reach_Individual"]):
            ax_bar.text(v + 0.01, i, f"{v:.2%}", va='center')
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Reach (%)")
        barras_buf = render_small_fig(fig_bar)
        st.image(barras_buf)
        st.download_button("⬇️ Descargar gráfico Barras como PNG", data=add_header_footer(barras_buf), file_name="barras_reach_individual.png", mime="image/png")

        # Gráfico Venn
        if len(df) == 3:
            st.markdown("---")
            st.markdown("### Overlap de medios")
            venn_fig, venn_ax = plt.subplots(figsize=(10, 5))
            venn = venn3(subsets=(0.4, 0.4, 0.3, 0.2, 0.1, 0.15, 0.05), set_labels=df["Medio"].tolist())
            for text in venn.set_labels + venn.subset_labels:
                if text:
                    try:
                        text.set_text(f"{float(text.get_text()):.2%}")
                    except:
                        pass
            venn_buf = render_small_fig(venn_fig)
            st.image(venn_buf)
            st.download_button("⬇️ Descargar gráfico Venn como PNG", data=add_header_footer(venn_buf), file_name="venn_overlap.png", mime="image/png")
        else:
            st.info("El gráfico de Venn se mostrará solo si hay exactamente tres medios.")

        # Descarga PDF
        st.markdown("---")
        st.markdown("### Descarga de reporte PDF")
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for image_buf in [curva_buf, barras_buf] + ([venn_buf] if len(df) == 3 else []):
                img = Image.open(add_header_footer(image_buf))
                fig_pdf = plt.figure(figsize=(img.width / 100, img.height / 100))
                ax_pdf = fig_pdf.add_axes([0, 0, 1, 1])
                ax_pdf.axis("off")
                ax_pdf.imshow(img)
                pdf.savefig(fig_pdf)
                plt.close(fig_pdf)
        pdf_buffer.seek(0)
        st.download_button("📄 Descargar reporte completo en PDF", data=pdf_buffer, file_name="report_reach_cume.pdf", mime="application/pdf")
    else:
        st.info("Revisa que todos los medios tengan nombre, valores válidos y no estén duplicados para generar el reporte.")

# Footer
st.markdown("""
    <div style="background-color:#000000; padding: 10px 30px; position: fixed; bottom: 0; left: 0; width: 100%; display: flex; justify-content: space-between; align-items: center; z-index: 100;">
        <div style="color: white; font-size: 14px;">
            © 2025 Effective &nbsp;&nbsp;|&nbsp;&nbsp; Designed by ArchimED
        </div>
        <div style="display: flex; gap: 10px;">
            <img src="https://effective.com.mx/wp-content/uploads/2024/10/logo-main-nav.png" height="30">
            <img src="https://firebasestorage.googleapis.com/v0/b/effective-ecad2.appspot.com/o/logos%20ereach%2Farchimed%20logo.png?alt=media&token=36551e78-097d-4372-ba1f-8520ec375b8c" height="40">
        </div>
    </div>
""", unsafe_allow_html=True)




