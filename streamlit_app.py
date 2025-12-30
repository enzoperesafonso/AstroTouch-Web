import time
import io
import os
import numpy as np
import streamlit as st
import plotly.express as px
from astropy.io import fits
from stl import mesh
from scipy.ndimage import gaussian_filter, zoom
from streamlit_stl import stl_from_file

# --- CONFIG ---
st.set_page_config(page_title='AstroTouch', layout="wide")

TEMP_STL_PATH = "astrotouch_output.stl"


# --- CORE TRANSFORMATION FUNCTION ---
def create_stl_from_fits(fits_input, stl_filepath, params):
    longest_side_mm = params.get('w', 100.0)
    max_height_mm = params.get('z', 10.0)
    base_thickness_mm = params.get('base', 2.0)
    invert = params.get('invert', False)
    log_scale = params.get('log', True)
    clip_percentile = params.get('clip', 1.0)
    smoothing_sigma = params.get('smooth', 0.5)
    downsample_factor = params.get('down', 1)
    border_width_mm = params.get('b_w', 0.0)
    border_height_mm = params.get('b_h', 0.0)

    with fits.open(fits_input) as hdul:
        # Auto-detect first HDU with data
        raw_data = None
        for hdu in hdul:
            if hdu.data is not None:
                raw_data = hdu.data
                break
        if raw_data is None:
            raise ValueError("No valid image data found in FITS extensions.")
        image_data = raw_data.astype(np.float32)

    if downsample_factor > 1:
        image_data = zoom(image_data, 1 / downsample_factor, order=1)

    image_data = np.nan_to_num(image_data, nan=np.nanmin(image_data) if np.any(np.isfinite(image_data)) else 0)

    if clip_percentile > 0:
        min_p = np.percentile(image_data, clip_percentile)
        max_p = np.percentile(image_data, 100 - clip_percentile)
        image_data = np.clip(image_data, min_p, max_p)

    if log_scale:
        image_data -= np.min(image_data)
        image_data = np.log1p(image_data)
    if invert:
        image_data = -image_data

    min_val, max_val = np.min(image_data), np.max(image_data)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1
    z_data = base_thickness_mm + ((image_data - min_val) / denom) * max_height_mm

    if smoothing_sigma > 0:
        z_data = gaussian_filter(z_data, sigma=smoothing_sigma)

    ny, nx = z_data.shape
    scale_factor = longest_side_mm / max(nx, ny)

    if border_width_mm > 0:
        bp = int(round(border_width_mm / scale_factor))
        z_bordered = np.full((ny + 2 * bp, nx + 2 * bp), base_thickness_mm + border_height_mm)
        z_bordered[bp:bp + ny, bp:bp + nx] = z_data
        z_data = z_bordered
        ny, nx = z_data.shape

    x = np.arange(nx) * scale_factor
    y = np.arange(ny) * scale_factor
    xx, yy = np.meshgrid(x, y)

    num_vertices = nx * ny
    vertices = np.zeros((num_vertices * 2, 3), dtype=np.float32)
    vertices[:num_vertices, 0] = xx.flatten()
    vertices[:num_vertices, 1] = yy.flatten()
    vertices[:num_vertices, 2] = z_data.flatten()
    vertices[num_vertices:, 0:2] = vertices[:num_vertices, 0:2]
    vertices[num_vertices:, 2] = 0.0

    num_faces_per_surface = 2 * (nx - 1) * (ny - 1)
    num_side_faces = 2 * (nx - 1) + 2 * (ny - 1)
    total_faces = (num_faces_per_surface * 2) + (num_side_faces * 2)
    faces = np.zeros((total_faces, 3), dtype=np.uint32)

    face_idx = 0
    base_offset = num_vertices

    for j in range(ny - 1):
        for i in range(nx - 1):
            v00, v10 = j * nx + i, j * nx + (i + 1)
            v01, v11 = (j + 1) * nx + i, (j + 1) * nx + (i + 1)
            faces[face_idx] = [v00, v10, v01];
            face_idx += 1
            faces[face_idx] = [v10, v11, v01];
            face_idx += 1
            faces[face_idx] = [base_offset + v00, base_offset + v01, base_offset + v10];
            face_idx += 1
            faces[face_idx] = [base_offset + v10, base_offset + v01, base_offset + v11];
            face_idx += 1

    for i in range(nx - 1):
        v_t0, v_t1 = i, i + 1
        v_b0, v_b1 = base_offset + i, base_offset + i + 1
        faces[face_idx] = [v_t0, v_b0, v_b1];
        face_idx += 1
        faces[face_idx] = [v_t0, v_b1, v_t1];
        face_idx += 1
        v_t0, v_t1 = (ny - 1) * nx + i, (ny - 1) * nx + i + 1
        v_b0, v_b1 = base_offset + (ny - 1) * nx + i, base_offset + (ny - 1) * nx + i + 1
        faces[face_idx] = [v_t0, v_b1, v_b0];
        face_idx += 1
        faces[face_idx] = [v_t0, v_t1, v_b1];
        face_idx += 1

    for j in range(ny - 1):
        v_t0, v_t1 = j * nx, (j + 1) * nx
        v_b0, v_b1 = base_offset + j * nx, base_offset + (j + 1) * nx
        faces[face_idx] = [v_t0, v_b1, v_b0];
        face_idx += 1
        faces[face_idx] = [v_t0, v_t1, v_b1];
        face_idx += 1
        v_t0, v_t1 = j * nx + (nx - 1), (j + 1) * nx + (nx - 1)
        v_b0, v_b1 = base_offset + j * nx + (nx - 1), base_offset + (j + 1) * nx + (nx - 1)
        faces[face_idx] = [v_t0, v_b0, v_b1];
        face_idx += 1
        faces[face_idx] = [v_t0, v_b1, v_t1];
        face_idx += 1

    out_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    out_mesh.vectors = vertices[faces]
    out_mesh.save(stl_filepath)
    return True


# --- SESSION STATE ---
if "file_bytes" not in st.session_state: st.session_state.file_bytes = None
if "model_ready" not in st.session_state: st.session_state.model_ready = False
if "log_on" not in st.session_state: st.session_state.log_on = False
if "invert_on" not in st.session_state: st.session_state.invert_on = False


# --- SIDEBAR ---
@st.fragment
def render_settings():
    # Make sure at_logo.png is in your project root
    st.logo("at_logo.png", size="large", link="https://github.com/enzoperesafonso/AstroTouch/tree/main")
    st.markdown(
        "Convert astronomical FITS images into 3D printable STL surface relief models.")

    # --- ADDED: SOURCE SELECTION ---
    source_type = st.radio("Select Data Source", ["Example", "Upload FITS"], horizontal=True)

    if source_type == "Upload FITS":
        file = st.file_uploader("Upload FITS", type=["fits", "fz"], help="Supports .fits and compressed .fz files.")
        if file:
            file.seek(0)
            new_bytes = file.read()
            if new_bytes != st.session_state.file_bytes:
                st.session_state.file_bytes = new_bytes
                st.session_state.model_ready = False
                st.rerun()
    else:
        # Load local example file
        example_path = "resorces/jupiter_Miri.fits"
        if os.path.exists(example_path):
            with open(example_path, "rb") as f:
                new_bytes = f.read()
                if new_bytes != st.session_state.file_bytes:
                    st.session_state.file_bytes = new_bytes
                    st.session_state.model_ready = False
                    st.rerun()
        else:
            st.error(f"Example file '{example_path}' not found.")

    st.subheader('Processing')
    nl = st.toggle("Log Scale", value=st.session_state.log_on,
                   help="Applies log(1+x) scaling to enhance faint details.")
    ni = st.toggle("Invert", value=st.session_state.invert_on, help="Inverts the height map.")

    if nl != st.session_state.log_on or ni != st.session_state.invert_on:
        st.session_state.log_on = nl
        st.session_state.invert_on = ni
        st.rerun()

    st.subheader('Geometry')
    z = st.slider('Relief Height (mm)', 1.0, 50.0, 10.0, help="Maximum height of features above the base plate.")
    w = st.slider('Model Width (mm)', 10.0, 300.0, 200.0, help="Scales the longest side of the model in mm.")
    bt = st.slider('Base Thickness (mm)', 1.0, 10.0, 2.0, help="Thickness of the bottom support plate.")

    st.subheader('Border')
    b_on = st.toggle('Add Border', value=True)
    bw = st.slider('Border Width (mm)', 0.1, 20.0, 5.0, disabled=not b_on)
    bh = st.slider('Border Height (mm)', 0.0, 20.0, 2.0, disabled=not b_on)

    st.subheader('Quality')
    clip = st.slider('Clipping %', 0.0, 5.0, 1.0, help="Removes outlier pixels (hot pixels/bright sources).")
    sm = st.slider('Smoothing', 0.0, 5.0, 2.0, help="Applies Gaussian blur to the mesh surface.")
    ds = st.slider('Downsample', 1, 10, 4, help="Reduces resolution for faster processing.")

    if st.button("Generate 3D Model", type="primary", width="stretch"):
        if st.session_state.file_bytes:
            st.session_state.params = {
                "z": z, "w": w, "base": bt, "log": st.session_state.log_on,
                "invert": st.session_state.invert_on, "clip": clip,
                "smooth": sm, "down": ds, "b_w": bw if b_on else 0, "b_h": bh if b_on else 0
            }
            st.session_state.model_ready = True
            st.rerun()


with st.sidebar:
    render_settings()

# --- MAIN PAGE ---
tab1, tab2, tab3 = st.tabs(["Fits Preview", "Model View", "About"])

if st.session_state.file_bytes:
    with tab1:
        st.subheader("Interactive FITS Preview")
        try:
            with fits.open(io.BytesIO(st.session_state.file_bytes)) as h:
                # Auto-find data for preview
                img_raw = None
                for hdu in h:
                    if hdu.data is not None:
                        img_raw = hdu.data
                        break
                if img_raw.ndim > 2: img_raw = img_raw[0]
                img = np.nan_to_num(img_raw)

            p_low, p_high = np.percentile(img, [1, 99])
            prev = np.clip(img, p_low, p_high)
            stride = max(1, prev.shape[0] // 1200, prev.shape[1] // 1200)
            prev = prev[::stride, ::stride].copy()

            if st.session_state.invert_on: prev = np.max(prev) - prev
            if st.session_state.log_on: prev = np.log10(prev - np.min(prev) + 1)

            fig = px.imshow(prev, origin='lower',
                            color_continuous_scale="viridis_r" if st.session_state.invert_on else "viridis")
            fig.update_layout(dragmode='pan', margin=dict(l=0, r=0, b=0, t=0), xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True})
        except Exception as e:
            st.error(f"Visualization error: {e}")

    with tab2:
        if st.session_state.model_ready:
            with st.spinner("Processing Mesh..."):
                create_stl_from_fits(io.BytesIO(st.session_state.file_bytes), TEMP_STL_PATH, st.session_state.params)
            if os.path.exists(TEMP_STL_PATH):
                stl_from_file(file_path=TEMP_STL_PATH, auto_rotate=True)
                st.success("Sucessfully generated STL file!")
                with open(TEMP_STL_PATH, "rb") as f:
                    st.download_button("Download STL", f, "astro_model.stl", width="stretch", type="primary")
        else:
            st.info("Click 'Generate' in the sidebar to create the mesh.")
else:
    with tab1:
        st.info("Upload a file or select 'Example' in the sidebar to begin.")
    with tab2:
        st.info("Upload a file and click 'Generate' to see the model.")

with tab3:
    st.title("About AstroTouch")

    st.markdown("""
        ### From Light to Touch
        **AstroTouch** is an open-source tool designed to make astronomical data accessible to everyone, 
        particularly the visually impaired community, by generating **tactile relief map** models that can be 3D printed with any consumer grade 3D-Printer.

        ### How it Works
        Most astronomical images are stored in the **FITS** (Flexible Image Transport System) format, 
        which contains scientific data rather than just simple pixels. AstroTouch processes this data 
        through a specific mathematical pipeline:

        1.  **Intensity Mapping**: We treat pixel brightness as physical height ($Z$-axis). 
            Higher intensity values become peaks, while darker background areas become valleys.
        2.  **Logarithmic Scaling**: Astronomical objects often have an extreme dynamic range. 
            We apply a logarithmic transform $Z_{new} = log(1+Z)$ to ensure faint structures 
            become tactilely perceptible.
        3.  **Mesh Generation**: The processed array is converted into a 3D manifold (STL), 
            connecting thousands of triangles into a watertight surface suitable for 3D printing.

        ### Open Source & Contribution
        AstroTouch is an ongoing effort to bridge the gap between science and accessibility. 
        To see the code behind this tool, report issues, or contribute to its development, please visit the **[GitHub repository](https://github.com/enzoperesafonso/AstroTouch/tree/main)**.

        ### Supporting Accessibility
        If you find this software useful and would like to support the cause of accessibility in South Africa, please consider making a small donation to the **South African Guide Dogs Association for the Blind**. 
        Your contribution helps provide mobility and independence to those in need.

        👉 **[Donate to SA Guide Dogs Association](https://guidedog.org.za/donate/)**

        ### Acknowledgements
        This tool relies heavily on [Astropy](https://www.astropy.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Streamlit-stl](https://github.com/Lucandia/streamlit_stl) and [NumPy-STL](https://github.com/WoLpH/numpy-stl/).
        """)

# --- FOOTER ---
st.markdown("---")
st.caption("🔭 **AstroTouch v1.0** | Created by **Enzo Peres Afonso**")