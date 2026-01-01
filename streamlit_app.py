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
st.set_page_config(
    page_title='AstroTouch | 3D FITS Converter',
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEMP_STL_PATH = "astrotouch_output.stl"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    div[st-decorator="true"] { display: none; }
    .stButton button { width: 100%; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "file_bytes" not in st.session_state: st.session_state.file_bytes = None
if "model_ready" not in st.session_state: st.session_state.model_ready = False
if "params" not in st.session_state:
    st.session_state.params = {
        "z": 10.0, "w": 200.0, "base": 2.0, "log": False,
        "invert": False, "clip": 1.0, "smooth": 2.0, "down": 4,
        "b_w": 5.0, "b_h": 2.0
    }


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
        image_data = np.max(image_data) - image_data

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

    # Generate Surfaces
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

    # Generate Side Walls
    for i in range(nx - 1):
        # Top edge
        v_t0, v_t1 = i, i + 1
        v_b0, v_b1 = base_offset + i, base_offset + i + 1
        faces[face_idx] = [v_t0, v_b0, v_b1];
        face_idx += 1
        faces[face_idx] = [v_t0, v_b1, v_t1];
        face_idx += 1
        # Bottom edge
        v_t0, v_t1 = (ny - 1) * nx + i, (ny - 1) * nx + i + 1
        v_b0, v_b1 = base_offset + (ny - 1) * nx + i, base_offset + (ny - 1) * nx + i + 1
        faces[face_idx] = [v_t0, v_b1, v_b0];
        face_idx += 1
        faces[face_idx] = [v_t0, v_t1, v_b1];
        face_idx += 1

    for j in range(ny - 1):
        # Left edge
        v_t0, v_t1 = j * nx, (j + 1) * nx
        v_b0, v_b1 = base_offset + j * nx, base_offset + (j + 1) * nx
        faces[face_idx] = [v_t0, v_b1, v_b0];
        face_idx += 1
        faces[face_idx] = [v_t0, v_t1, v_b1];
        face_idx += 1
        # Right edge
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


# --- SIDEBAR ---
with st.sidebar:
    st.logo("resorces/at_logo.png", size="large", link="https://github.com/enzoperesafonso/AstroTouch")
    st.markdown("Convert astronomical FITS images into 3D printable models.")

    source_type = st.radio("Select Data Source", ["Example", "Upload FITS"], horizontal=True)

    if source_type == "Upload FITS":
        file = st.file_uploader("Upload FITS", type=["fits", "fz"])
        if file:
            st.session_state.file_bytes = file.getvalue()
    else:
        example_path = "resorces/jupiter_Miri.fits"
        if os.path.exists(example_path):
            with open(example_path, "rb") as f:
                st.session_state.file_bytes = f.read()


    with st.form("settings_form"):
        with st.expander("Processing Options", expanded=True):
            log_scale = st.toggle("Logarithmic Scaling", value=st.session_state.params["log"], help="Applies log(1+x) scaling to enhance faint details.")
            invert = st.toggle("Invert Values", value=st.session_state.params["invert"], help="Inverts the height map. For the model valleys become hills and vice versa")
            clip = st.slider('Clipping %', 0.0, 5.0, st.session_state.params["clip"], help="Clips the lowest and highest percentile of pixels. Helps with hot pixels or saturated sources.", key="clip")
            sm = st.slider('Smoothing', 0.0, 5.0, st.session_state.params["smooth"], help="Applies Gaussian smoothing. A value of 1.0 to 2.0 is recommended for tactile models.", key="smooth")
            ds = st.select_slider('Downsample Factor', options=[1, 2, 4, 8, 16], value=st.session_state.params["down"], help="Reduces image resolution by this factor. This helps to speed up processing as well as reduce the total storage size of the model.", key="down")

        with st.expander("Model Dimensions", expanded=False):
            w = st.slider('Width (mm)', 10.0, 300.0, st.session_state.params["w"], help="Scales the image content so its longest side matches this value in mm. The final model will be larger if a border is added.", key="width")
            z = st.slider('Relief Height (mm)', 1.0, 50.0, st.session_state.params["z"], help="Maximum height of the features above the base in mm.", key="height")
            bt = st.slider('Base Thickness (mm)', 1.0, 10.0, st.session_state.params["base"], help="Thickness of the model's base in mm.")

        with st.expander("Border Settings", expanded=False):
            b_on = st.checkbox('Add Border', value=True)
            bw = st.slider('Border Width', 0.1, 20.0, st.session_state.params["b_w"], help="Width of border border in mm.")
            bh = st.slider('Border Height', 0.0, 10.0, st.session_state.params["b_h"], help="Sets the height of the border, measured from the base in mm.")


        submitted = st.form_submit_button("🔨 Generate 3D Model", type="primary")
        if submitted:
            st.session_state.params = {
                "z": z, "w": w, "base": bt, "log": log_scale, "invert": invert,
                "clip": clip, "smooth": sm, "down": ds,
                "b_w": bw if b_on else 0, "b_h": bh if b_on else 0
            }
            st.session_state.model_ready = True

# --- MAIN PAGE ---
tab1, tab2, tab3 = st.tabs(["Fits Preview", "Model View", "About"])

if st.session_state.file_bytes:
    with tab1:
        col_img, col_metrics = st.columns([3, 1])
        with col_img:
            with st.spinner("Analyzing FITS Data..."):
                with fits.open(io.BytesIO(st.session_state.file_bytes)) as h:
                    img_raw = next(hdu.data for hdu in h if hdu.data is not None)
                    if img_raw.ndim > 2: img_raw = img_raw[0]
                    img = np.nan_to_num(img_raw)

                p_low, p_high = np.percentile(img, [1, 99])
                prev = np.clip(img, p_low, p_high)

                # Dynamic visual preview based on logic toggles
                if st.session_state.params["invert"]: prev = np.max(prev) - prev
                if st.session_state.params["log"]: prev = np.log1p(prev - np.min(prev))

                fig = px.imshow(prev, color_continuous_scale="viridis", origin='lower')
                fig.update_layout(dragmode='pan', margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        with col_metrics:
            st.subheader("Fits Info")
            st.metric("Native Resolution", f"{img.shape[1]}x{img.shape[0]}")
            st.metric("Mesh Density",
                      f"~{(img.shape[0] // st.session_state.params['down']) * (img.shape[1] // st.session_state.params['down'])} pts")
            st.info("Log scaling is recommended for nebulae and galaxies to make faint gas clouds tactile.")

    with tab2:
        if st.session_state.model_ready:
            status = st.status("Constructing Mesh...", expanded=True)
            status.write("Calculating normals and vertex heights...")
            create_stl_from_fits(io.BytesIO(st.session_state.file_bytes), TEMP_STL_PATH, st.session_state.params)
            status.write("Generating side walls and base...")
            time.sleep(1)  # Visual pause for UX
            status.update(label="3D Model Created!", state="complete", expanded=False)

            c1, c2 = st.columns([2, 1])
            with c1:
                stl_from_file(file_path=TEMP_STL_PATH, auto_rotate=True)
            with c2:
                st.subheader("Ready for Printing")
                st.markdown("""
                                <div class="accessibility-box">
                                <b>1.</b> Use <b>Log Scaling</b>. In many FITS images, stars are bright but galaxies are faint. Log scaling lifts the "tactile floor" so the faint structures aren't lost in the baseplate.
                                <br><br>
                                <b>2.</b> Keep <b>Smoothing</b> above 1.5. High-frequency digital noise feels like grit to the fingers. A smoother surface allows the user to feel the overall shape of the object.
                                <br><br>
                                <b>3.</b> A 10-15mm relief height is usually the "sweet spot" for tactile exploration. Any higher can make vertical walls sharp or fragile.
                                <br><br>
                                </div>
                                """, unsafe_allow_html=True)
                with open(TEMP_STL_PATH, "rb") as f:
                    st.download_button("Download STL File", f, "astro_model.stl", type="primary")
                st.caption("Recommended: 0.15mm - 0.2mm layer height.")
        else:
            st.info("Click 'Generate 3D Model' in the sidebar to build the mesh.")

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

st.caption("🔭 AstroTouch v1.1 | Developed by Enzo Peres Afonso")