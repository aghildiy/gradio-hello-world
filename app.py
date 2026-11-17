# --- diagnostics bootstrap: install crash/exception logging before app code ---
import os as _diag_os
import sys as _diag_sys

_diag_dir = _diag_os.path.dirname(_diag_os.path.abspath(__file__))
if _diag_dir not in _diag_sys.path:
    _diag_sys.path.insert(0, _diag_dir)
import app_diagnostics  # noqa: E402

app_diagnostics.install_diagnostics()
# --- end diagnostics bootstrap ---

import logging
import os

import gradio as gr
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_UPLOAD_ROWS = 100_000
MAX_UPLOAD_SIZE = '10mb'

FUNCTIONS = {
    'Exponential (2^x)': lambda x: np.power(2.0, x),
    'Quadratic (x^2)': lambda x: x ** 2,
    'Linear (x)': lambda x: x,
    'Sine (sin x)': np.sin,
}

THEMES = {
    'Soft': gr.themes.Soft,
    'Default': gr.themes.Default,
    'Glass': gr.themes.Glass,
    'Monochrome': gr.themes.Monochrome,
    'Ocean': gr.themes.Ocean,
}

CSS = """
footer {visibility: hidden}
#title {text-align: center}
"""

# Reload with Gradio's ?__theme= param so charts (which pick colors on mount) re-theme too.
TOGGLE_DARK_JS = """() => {
    const url = new URL(window.location.href);
    const isDark = document.body.classList.contains('dark');
    url.searchParams.set('__theme', isDark ? 'light' : 'dark');
    window.location.replace(url.toString());
}"""


def generate_data(func_name: str, x_min: float, x_max: float,
                  points: int, log_y: bool) -> pd.DataFrame:
    if x_min >= x_max:
        raise gr.Error('X min must be less than X max.')
    x = np.linspace(x_min, x_max, int(points))
    y = FUNCTIONS[func_name](x)
    if log_y:
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log10(y)
    df = pd.DataFrame({'x': x, 'y': y})
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def update_explorer(func_name, x_min, x_max, points, log_y, chart_type):
    df = generate_data(func_name, x_min, x_max, points, log_y)
    y_title = f"log10({func_name})" if log_y else func_name
    is_line = chart_type == 'Line'
    return (
        gr.ScatterPlot(value=df, y_title=y_title, visible=not is_line),
        gr.LinePlot(value=df, y_title=y_title, visible=is_line),
        df,
    )


def load_upload(file_path):
    if not file_path:
        empty = gr.Dropdown(choices=[], value=None)
        return None, empty, empty, 'Upload a CSV file to begin.'
    try:
        df = pd.read_csv(file_path, nrows=MAX_UPLOAD_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        logger.warning('Failed to parse uploaded CSV: %s', type(exc).__name__)
        raise gr.Error('Could not read the file. Please upload a valid UTF-8 CSV.')

    note = f'Loaded {len(df):,} rows x {len(df.columns)} columns.'
    if len(df) > MAX_UPLOAD_ROWS:
        df = df.head(MAX_UPLOAD_ROWS)
        note = f'File truncated to the first {MAX_UPLOAD_ROWS:,} rows.'

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.astype(str).tolist()
    df.columns = all_cols
    if not numeric_cols:
        note += ' No numeric columns found for the Y axis.'
    x_default = all_cols[0] if all_cols else None
    y_default = next((c for c in numeric_cols if c != x_default),
                     numeric_cols[0] if numeric_cols else None)
    return (
        df,
        gr.Dropdown(choices=all_cols, value=x_default),
        gr.Dropdown(choices=[str(c) for c in numeric_cols], value=y_default),
        note,
    )


def summarize(df):
    if df is None:
        return None
    numeric = df.select_dtypes(include='number')
    if numeric.empty:
        return pd.DataFrame({'info': ['No numeric columns to summarize.']})
    stats = numeric.describe().T
    stats.insert(0, 'missing', numeric.isna().sum())
    return stats.round(4).reset_index(names='column')


def plot_upload(df, x_col, y_col, chart_type):
    if df is None or not x_col or not y_col:
        return gr.ScatterPlot(value=None), gr.LinePlot(value=None, visible=False)
    plot_df = df[[x_col, y_col]].dropna() if x_col != y_col else df[[x_col]].dropna()
    is_line = chart_type == 'Line'
    return (
        gr.ScatterPlot(value=plot_df, x=x_col, y=y_col,
                       x_title=x_col, y_title=y_col, visible=not is_line),
        gr.LinePlot(value=plot_df, x=x_col, y=y_col,
                    x_title=x_col, y_title=y_col, visible=is_line),
    )


with gr.Blocks(title='Hello world') as gradio_app:
    with gr.Row():
        gr.Markdown('# Hello world!', elem_id='title')
    with gr.Row():
        dark_btn = gr.Button('Toggle dark mode', size='sm', scale=0)
    dark_btn.click(None, js=TOGGLE_DARK_JS)

    with gr.Tabs():
        with gr.Tab('Explore'):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    func = gr.Dropdown(list(FUNCTIONS), value='Exponential (2^x)',
                                       label='Function')
                    x_min = gr.Number(value=0, label='X min')
                    x_max = gr.Number(value=29, label='X max')
                    points = gr.Slider(5, 500, value=30, step=1, label='Points')
                    log_y = gr.Checkbox(label='Log10 scale for Y')
                    chart = gr.Radio(['Scatter', 'Line'], value='Scatter',
                                     label='Chart type')
                with gr.Column(scale=3):
                    initial = generate_data('Exponential (2^x)', 0, 29, 30, False)
                    scatter = gr.ScatterPlot(value=initial, x='x', y='y', height=400,
                                             x_title='x', y_title='Exponential (2^x)')
                    line = gr.LinePlot(value=initial, x='x', y='y', height=400,
                                       x_title='x', y_title='Exponential (2^x)',
                                       visible=False)
                    table = gr.Dataframe(value=initial, label='Data',
                                         max_height=250, interactive=False)

            explorer_inputs = [func, x_min, x_max, points, log_y, chart]
            gr.on(
                triggers=[c.change for c in explorer_inputs],
                fn=update_explorer,
                inputs=explorer_inputs,
                outputs=[scatter, line, table],
                show_progress='hidden',
            )

        with gr.Tab('Upload data'):
            upload_state = gr.State()
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    upload = gr.File(label=f'CSV file (max {MAX_UPLOAD_SIZE})',
                                     file_types=['.csv'], type='filepath')
                    status = gr.Markdown('Upload a CSV file to begin.')
                    x_col = gr.Dropdown(label='X column', choices=[])
                    y_col = gr.Dropdown(label='Y column (numeric)', choices=[])
                    up_chart = gr.Radio(['Scatter', 'Line'], value='Scatter',
                                        label='Chart type')
                with gr.Column(scale=3):
                    up_scatter = gr.ScatterPlot(height=400)
                    up_line = gr.LinePlot(height=400, visible=False)
                    preview = gr.Dataframe(label='Preview', max_height=250,
                                           interactive=False)
                    with gr.Accordion('Summary statistics', open=False):
                        summary = gr.Dataframe(interactive=False, max_height=250)

            upload.change(
                load_upload, inputs=upload,
                outputs=[upload_state, x_col, y_col, status],
            ).then(
                lambda df: (df.head(100) if df is not None else None, summarize(df)),
                inputs=upload_state, outputs=[preview, summary],
            )
            gr.on(
                triggers=[x_col.change, y_col.change, up_chart.change],
                fn=plot_upload,
                inputs=[upload_state, x_col, y_col, up_chart],
                outputs=[up_scatter, up_line],
            )

if __name__ == '__main__':
    theme_name = os.getenv('GRADIO_THEME', 'Soft')
    gradio_app.launch(
        theme=THEMES.get(theme_name, gr.themes.Soft)(),
        css=CSS,
        max_file_size=MAX_UPLOAD_SIZE,
        server_name=os.getenv('GRADIO_SERVER_NAME', '0.0.0.0'),
    )
