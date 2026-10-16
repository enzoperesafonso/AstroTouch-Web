import hashlib
import io
import os
import tempfile

import numpy as np
import plotly.express as px
import streamlit as st
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from stl import mesh

from streamlit_stl import stl_from_file

# --- CONFIG ---
st.set_page_config(
    page_title='AstroTouch | 3D FITS Converter',
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXAMPLE_PATH = "resorces/jupiter_Miri.fits"
MAX_GRID_POINTS = 2_000_000   # guard against meshes that exhaust server memory
PREVIEW_MAX_SIDE = 1000       # longest side of the 2D preview, in pixels

DEFAULT_PARAMS = {
    "z": 10.0, "w": 200.0, "base": 2.0, "log": False,
    "invert": False, "clip": 1.0, "smooth": 2.0, "down": 4,
    "b_w": 5.0, "b_h": 2.0,
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
st.session_state.setdefault("file_bytes", None)
st.session_state.setdefault("file_id", None)
st.session_state.setdefault("stl_path", None)
st.session_state.setdefault("stl_size", 0)


# --- FITS HELPERS ---
def find_image(hdul):
    """Return the first HDU holding a numeric array with at least two dimensions as a 2D float32 array."""
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if not isinstance(data, np.ndarray) or data.dtype.names is not None:
            continue  # empty HDU or binary/ASCII table
        if data.ndim < 2 or data.dtype.kind not in "iuf":
            continue
        data = np.asarray(data, dtype=np.float32)
        while data.ndim > 2:  # data cubes: use the first plane
            data = data[0]
        if data.shape[0] > 1 and data.shape[1] > 1:
            return data
    raise ValueError("No 2D image data found in this FITS file.")


def clean_image(img):
    """Replace NaN/Inf with the minimum finite value."""
    finite = np.isfinite(img)
    if not finite.any():
        raise ValueError("The FITS image contains no finite pixel values.")
    if finite.all():
        return img
    out = img.copy()
    out[~finite] = img[finite].min()
    return out


@st.cache_data(show_spinner=False, max_entries=3)
def load_fits(file_bytes):
    with fits.open(io.BytesIO(file_bytes), memmap=False) as hdul:
        return clean_image(find_image(hdul))


def block_downsample(img, factor):
    """Downsample by averaging factor x factor blocks (edges that do not fit are trimmed)."""
    if factor <= 1:
        return img
    ny, nx = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    if ny == 0 or nx == 0:
        raise ValueError("Downsample factor is larger than the image.")
    blocks = img[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def normalize_image(img, clip, log_scale, invert):
    """Clip, optionally log-scale and invert, then normalise to 0..1. Shared by preview and mesh."""
    img = img.astype(np.float32, copy=True)
    if clip > 0:
        lo, hi = np.percentile(img, [clip, 100 - clip])
        img = np.clip(img, lo, hi)
    if log_scale:
        img -= img.min()
        img = np.log1p(img)
    if invert:
        img = img.max() - img
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


@st.cache_data(show_spinner=False, max_entries=8)
def make_preview(file_bytes, clip, log_scale, invert):
    img = load_fits(file_bytes)
    step = max(1, int(np.ceil(max(img.shape) / PREVIEW_MAX_SIDE)))
    return normalize_image(block_downsample(img, step), clip, log_scale, invert)


def grid_shape(img_shape, params):
    """Size of the final vertex grid (after downsampling and border)."""
    d = params["down"]
    ny, nx = img_shape[0] // d, img_shape[1] // d
    if params["b_w"] > 0 and nx > 0 and ny > 0:
        bp = int(round(params["b_w"] / (params["w"] / max(nx, ny))))
        ny, nx = ny + 2 * bp, nx + 2 * bp
    return ny, nx


# --- CORE TRANSFORMATION FUNCTION ---
def build_faces(ny, nx):
    """Triangles for a watertight slab: top surface, bottom surface and four side walls.
    Vertices [0, n) are the top grid; [n, 2n) are the matching bottom grid."""
    n = nx * ny
    idx = np.arange(n, dtype=np.uint32).reshape(ny, nx)
    v00, v10 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    v01, v11 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    o = np.uint32(n)

    parts = [
        np.stack([v00, v10, v01], axis=1),
        np.stack([v10, v11, v01], axis=1),
        np.stack([o + v00, o + v01, o + v10], axis=1),
        np.stack([o + v10, o + v01, o + v11], axis=1),
    ]

    def wall(edge, flip):
        t0, t1 = edge[:-1], edge[1:]
        b0, b1 = o + t0, o + t1
        if flip:
            return [np.stack([t0, b1, b0], axis=1), np.stack([t0, t1, b1], axis=1)]
        return [np.stack([t0, b0, b1], axis=1), np.stack([t0, b1, t1], axis=1)]

    parts += wall(idx[0, :], False)    # front (y = 0)
    parts += wall(idx[-1, :], True)    # back
    parts += wall(idx[:, 0], True)     # left
    parts += wall(idx[:, -1], False)   # right
    return np.concatenate(parts, axis=0)


def create_stl_from_fits(file_bytes, stl_filepath, params):
    img = block_downsample(load_fits(file_bytes), params["down"])
    if img.shape[0] < 2 or img.shape[1] < 2:
        raise ValueError("Image is too small after downsampling; use a lower downsample factor.")
    ny, nx = grid_shape(load_fits(file_bytes).shape, params)
    if nx * ny > MAX_GRID_POINTS:
        raise ValueError(
            f"The model would have {nx * ny:,} grid points (limit {MAX_GRID_POINTS:,}). "
            "Increase the downsample factor.")

    z_data = params["base"] + normalize_image(
        img, params["clip"], params["log"], params["invert"]) * params["z"]

    if params["smooth"] > 0:
        z_data = gaussian_filter(z_data, sigma=params["smooth"])

    ny, nx = z_data.shape
    scale_factor = params["w"] / max(nx, ny)

    if params["b_w"] > 0:
        bp = int(round(params["b_w"] / scale_factor))
        if bp > 0:
            bordered = np.full((ny + 2 * bp, nx + 2 * bp), params["base"] + params["b_h"], dtype=np.float32)
            bordered[bp:bp + ny, bp:bp + nx] = z_data
            z_data = bordered
            ny, nx = z_data.shape

    xx, yy = np.meshgrid(np.arange(nx) * scale_factor, np.arange(ny) * scale_factor)
    n = nx * ny
    vertices = np.zeros((n * 2, 3), dtype=np.float32)
    vertices[:n, 0] = xx.ravel()
    vertices[:n, 1] = yy.ravel()
    vertices[:n, 2] = z_data.ravel()
    vertices[n:, :2] = vertices[:n, :2]

    faces = build_faces(ny, nx)
    out_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    out_mesh.vectors = vertices[faces]
    out_mesh.update_normals()
    out_mesh.save(stl_filepath)


def set_source(data, name):
    """Store the active FITS file; drop any model built from a previous file."""
    file_id = hashlib.md5(data).hexdigest() if data else None
    if file_id != st.session_state.file_id:
        st.session_state.file_id = file_id
        st.session_state.file_bytes = data
        st.session_state.file_name = name
        clear_model()


def clear_model():
    path = st.session_state.stl_path
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    st.session_state.stl_path = None


# --- SIDEBAR ---
with st.sidebar:
    st.logo("resorces/at_logo.png", size="large", link="https://github.com/enzoperesafonso/AstroTouch")
    st.markdown("Convert astronomical FITS images into 3D printable models.")

    source_type = st.radio("Select Data Source", ["Example", "Upload FITS"], horizontal=True)

    if source_type == "Upload FITS":
        file = st.file_uploader("Upload FITS", type=["fits", "fit", "fts", "fz"])
        if file:
            set_source(file.getvalue(), file.name)
        else:
            set_source(None, None)
    elif os.path.exists(EXAMPLE_PATH):
        with open(EXAMPLE_PATH, "rb") as f:
            set_source(f.read(), os.path.basename(EXAMPLE_PATH))
    else:
        st.error("Example file is missing.")
        set_source(None, None)

    D = DEFAULT_PARAMS
    with st.expander("Processing Options", expanded=True):
        log_scale = st.toggle("Logarithmic Scaling", value=D["log"], help="Applies log(1+x) scaling to enhance faint details.")
        invert = st.toggle("Invert Values", value=D["invert"], help="Inverts the height map. For the model valleys become hills and vice versa")
        clip = st.slider('Clipping %', 0.0, 5.0, D["clip"], step=0.1, help="Clips the lowest and highest percentile of pixels. Helps with hot pixels or saturated sources.")
        sm = st.slider('Smoothing', 0.0, 5.0, D["smooth"], step=0.1, help="Applies Gaussian smoothing. A value of 1.0 to 2.0 is recommended for tactile models.")
        ds = st.select_slider('Downsample Factor', options=[1, 2, 4, 8, 16], value=D["down"], help="Reduces image resolution by this factor. This helps to speed up processing as well as reduce the total storage size of the model.")

    with st.expander("Model Dimensions", expanded=False):
        w = st.slider('Width (mm)', 10.0, 300.0, D["w"], help="Scales the image content so its longest side matches this value in mm. The final model will be larger if a border is added.")
        z = st.slider('Relief Height (mm)', 1.0, 50.0, D["z"], help="Maximum height of the features above the base in mm.")
        bt = st.slider('Base Thickness (mm)', 1.0, 10.0, D["base"], help="Thickness of the model's base in mm.")

    with st.expander("Border Settings", expanded=False):
        b_on = st.checkbox('Add Border', value=True)
        bw = st.slider('Border Width (mm)', 0.1, 20.0, D["b_w"], help="Width of the border in mm.", disabled=not b_on)
        bh = st.slider('Border Height (mm)', 0.0, 10.0, D["b_h"], help="Sets the height of the border, measured from the base in mm.", disabled=not b_on)

    params = {
        "z": z, "w": w, "base": bt, "log": log_scale, "invert": invert,
        "clip": clip, "smooth": sm, "down": ds,
        "b_w": bw if b_on else 0.0, "b_h": bh if b_on else 0.0,
    }
    generate = st.button("🔨 Generate 3D Model", type="primary", disabled=st.session_state.file_bytes is None)

# --- GENERATE MODEL ---
build_error = None
if generate and st.session_state.file_bytes:
    clear_model()
    try:
        with st.spinner("Constructing mesh..."):
            fd, path = tempfile.mkstemp(suffix=".stl", prefix="astrotouch_")
            os.close(fd)
            st.session_state.stl_path = path
            create_stl_from_fits(st.session_state.file_bytes, path, params)
            st.session_state.stl_size = os.path.getsize(path)
            st.session_state.stl_params = dict(params)
        st.toast("3D model created! Open the Model View tab.", icon="✅")
    except Exception as e:
        clear_model()
        build_error = f"Could not build the model: {e}"

# --- MAIN PAGE ---
tab1, tab2, tab3 = st.tabs(["Fits Preview", "Model View", "About"])

with tab1:
    if st.session_state.file_bytes is None:
        st.info("Upload a FITS file in the sidebar to get started.")
    else:
        try:
            img = load_fits(st.session_state.file_bytes)
        except Exception as e:
            st.error(f"Could not read this FITS file: {e}")
            img = None
        if img is not None:
            col_img, col_metrics = st.columns([3, 1])
            with col_img:
                prev = make_preview(st.session_state.file_bytes, clip, log_scale, invert)
                fig = px.imshow(prev, color_continuous_scale="viridis", origin='lower')
                fig.update_layout(dragmode='pan', margin=dict(l=0, r=0, b=0, t=0), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch", config={'scrollZoom': True})

            with col_metrics:
                st.subheader("Fits Info")
                st.metric("Native Resolution", f"{img.shape[1]}x{img.shape[0]}")
                gy, gx = grid_shape(img.shape, params)
                st.metric("Mesh Density", f"~{gx * gy:,} pts")
                if gx * gy > MAX_GRID_POINTS:
                    st.warning("Too many points: increase the downsample factor.")
                st.info("Log scaling is recommended for nebulae and galaxies to make faint gas clouds tactile.")

with tab2:
    if build_error:
        st.error(build_error)
    elif st.session_state.stl_path and os.path.exists(st.session_state.stl_path):
        c1, c2 = st.columns([2, 1])
        with c1:
            stl_from_file(file_path=st.session_state.stl_path, auto_rotate=True, key="stl_viewer")
        with c2:
            st.subheader("Ready for Printing")
            st.markdown("""
**1.** Use **Log Scaling**. In many FITS images, stars are bright but galaxies are faint. Log scaling lifts the "tactile floor" so the faint structures aren't lost in the baseplate.

**2.** Keep **Smoothing** above 1.5. High-frequency digital noise feels like grit to the fingers. A smoother surface allows the user to feel the overall shape of the object.

**3.** A 10-15mm relief height is usually the "sweet spot" for tactile exploration. Any higher can make vertical walls sharp or fragile.
""")
            with open(st.session_state.stl_path, "rb") as f:
                stl_bytes = f.read()
            base_name = os.path.splitext(st.session_state.get("file_name") or "astro")[0]
            st.download_button("Download STL File", stl_bytes, f"{base_name}_astrotouch.stl",
                               mime="model/stl", type="primary")
            st.caption(f"File size: {st.session_state.stl_size / 1e6:.1f} MB. Recommended: 0.15mm - 0.2mm layer height.")
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
        To see the code behind this tool, report issues, or contribute to its development, please visit the **[GitHub repository](https://github.com/enzoperesafonso/AstroTouch-Web)**.

        ### Supporting Accessibility
        If you find this software useful and would like to support the cause of accessibility in South Africa, please consider making a small donation to the **South African Guide Dogs Association for the Blind**. 
        Your contribution helps provide mobility and independence to those in need.

        👉 **[Donate to SA Guide Dogs Association](https://guidedog.org.za/donate/)**

        ### Acknowledgements
        This tool relies heavily on [Astropy](https://www.astropy.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Streamlit-stl](https://github.com/Lucandia/streamlit_stl) and [NumPy-STL](https://github.com/WoLpH/numpy-stl/).
        """)

st.caption("🔭 AstroTouch v1.1 | Developed by Enzo Peres Afonso")