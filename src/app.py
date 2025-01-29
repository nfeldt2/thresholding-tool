# src/app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import numpy as np
import SimpleITK as sitk
import os
from uuid import uuid4
import dash_bootstrap_components as dbc
import json
import functools

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --------------------------------------------------------------------------
# Global variables and data
current_scan_id = None
ct_volume = None
lung_mask = None
mask_itk = None
z_dim = None
y_dim = None
x_dim = None
modified_mask_global = None
previous_trigger = None
dicom_IDs = []

region_colors = [
    [0.5, 0.5, 0],   # Dark yellowish
    [0, 1, 1],       # Cyan
    [1, 0.5, 0],     # Orange
    [0.5, 0, 0.5],   # Purple
    [0, 0.5, 0.5],   # Teal
]
default_region_threshold = -720

def log_callback(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Retrieve the current callback context
        ctx = dash.callback_context

        if not ctx.triggered:
            trigger = "No trigger"
        else:
            # Get the ID of the component that triggered the callback
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # Log the callback invocation details
        print(f"\n[Callback Triggered] Function: {func.__name__}")
        print(f"Triggered by: {trigger}")
        print(f"Inputs: {args}")
        print(f"States: {kwargs}")

        # Execute the original callback function
        result = func(*args, **kwargs)

        print(f"[Callback Completed] Function: {func.__name__}\n")
        return result
    return wrapper

def get_next_region_color(num_regions):
    idx = num_regions % len(region_colors)
    return region_colors[idx]

def load_scan_data(scan_id):
    global ct_volume, lung_mask, mask_itk, z_dim, y_dim, x_dim, current_scan_id, modified_mask_global, dicom_IDs
    if scan_id == current_scan_id:
        return

    ct_volume_path = f'../CTA_Nathan/'
    lung_mask_path = f'../Lung_Masks/'

    #check if the ct_volume candidates are dicom or nifti
    if os.path.exists(ct_volume_path+scan_id+'.nii.gz'):
        ct_volume_local = sitk.GetArrayFromImage(sitk.ReadImage(ct_volume_path+scan_id+'.nii.gz'))
    elif os.path.exists(ct_volume_path+scan_id):
        ct_volume_path = ct_volume_path+scan_id
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_volume_path)
        reader.SetFileNames(dicom_names)
        full_volume_itk = reader.Execute()
        ct_volume_local = sitk.GetArrayFromImage(full_volume_itk)
        if dicom_IDs is None:
            # get only number value from the dicom_IDs
            dicom_IDs = [int(''.join(filter(str.isdigit, ct_volume_path)))]
        else:
            dicom_IDs.append(int(''.join(filter(str.isdigit,ct_volume_path))))

    # how do you check if a file is there?
    # check if the file exists
    if os.path.isfile(lung_mask_path+scan_id+'_mask.nii.gz'):
        print('occured')
        lung_mask_local = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path+scan_id+'_mask.nii.gz'))
        mask_itk_local = sitk.GetImageFromArray(lung_mask_local)
    if os.path.isfile(lung_mask_path+scan_id+'_lungsegm.nii.gz'):
        lung_mask_path = lung_mask_path+scan_id+'_lungsegm.nii.gz'
        lung_mask_local = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path))
        mask_itk_local = sitk.GetImageFromArray(lung_mask_local)

    ct_volume = ct_volume_local
    lung_mask = lung_mask_local
    mask_itk = mask_itk_local
    z_dim, y_dim, x_dim = ct_volume.shape
    current_scan_id = scan_id
    modified_mask_global = None
    previous_region_thresholds = None

# input your own path to the data
def get_available_scans():
    base_dir = '../CTA_Nathan'
    candidates = [d for d in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, d)) or os.path.isdir(os.path.join(base_dir, d))]
    cleaned_names = list(set(''.join(char for char in filename if char.isdigit()) for filename in candidates))
    if '' in cleaned_names:
        cleaned_names.remove('')
    print(candidates)
    print(cleaned_names)

    scans = []
    for c in cleaned_names:
        mask_path = f'../Lung_Masks/{c}_mask.nii.gz'
        alter_mask_path = f'../Lung_Masks/{c}_lungsegm.nii.gz'
        # list dir of ../lung_masks

        if os.path.exists(mask_path):
            scans.append(c)
        elif os.path.exists(alter_mask_path):
            scans.append(c)
    try:
        scans_sorted = sorted(scans, key=lambda x: int(x))
    except ValueError:
        scans_sorted = sorted(scans)
    
    print(scans)
    return scans_sorted

available_scans = get_available_scans()
if not available_scans:
    raise RuntimeError("No scans found.")

app.layout = html.Div([
    html.H1('Lung Lobe Threshold Adjustment'),


    html.Div([
        html.Label('Select Scan'),
        dcc.Dropdown(
            id='scan-selection',
            options=[{'label': s, 'value': s} for s in available_scans],
            value=available_scans[0],
            clearable=False
        ),
    ], style={'margin-top': '10px'}),
    dcc.Graph(id='ct-image'),

    # html.Div([
    #    dcc.Checklist(
    #        id='enable-drawing',
    #        options=[{'label': 'Enable Region Drawing', 'value': 'enable'}],
    #        value=[]
    #    ),
    #    dcc.Graph(id='ct-image', config={'modeBarButtonsToAdd': ['drawrect']}),
    #], style={'margin-top': '10px'}),

    html.Div([
        html.Label('View'),
        dcc.Dropdown(
            id='view-selection',
            options=[
                {'label': 'Axial', 'value': 'axial'},
                {'label': 'Coronal', 'value': 'coronal'},
                {'label': 'Sagittal', 'value': 'sagittal'}
            ],
            value='axial',
            clearable=False
        ),
    ], style={'margin-top': '10px'}),

    html.Div([
        html.Label('Slice Index'),
        dcc.Slider(
            id='slice-index',
            min=0,
            max=0,
            step=1,
            value=0,
            marks={},
            updatemode='drag',
        ),
    ], style={'margin-top': '10px'}),

    # Lobe thresholds
    html.Div([
        html.Label('Left Upper Lobe Threshold'),
        dcc.Input(
            id='threshold-lul',
            type='number',
            min=-1000,
            max=0,
            step=5,
            value=-720,
        ),
    ], style={'margin-top': '10px'}),

    html.Div([
        html.Label('Left Lower Lobe Threshold'),
        dcc.Input(
            id='threshold-lll',
            type='number',
            min=-1000,
            max=0,
            step=5,
            value=-720,
        ),
    ], style={'margin-top': '10px'}),

    html.Div([
        html.Label('Right Upper Lobe Threshold'),
        dcc.Input(
            id='threshold-rul',
            type='number',
            min=-1000,
            max=0,
            step=5,
            value=-720,
        ),
    ], style={'margin-top': '10px'}),

    html.Div([
        html.Label('Right Middle Lobe Threshold'),
        dcc.Input(
            id='threshold-rml',
            type='number',
            min=-1000,
            max=0,
            step=5,
            value=-720,
        ),
    ], style={'margin-top': '10px'}),

    html.Div([
        html.Label('Right Lower Lobe Threshold'),
        dcc.Input(
            id='threshold-rll',
            type='number',
            min=-1000,
            max=0,
            step=5,
            value=-720,
        ),
    ], style={'margin-top': '10px'}),

    # User-defined region thresholds container
    #html.Div(id='user-regions-container', style={'margin-top': '20px'}),

    # Save button and message
    html.Button('Save Mask', id='save-button', n_clicks=0, style={'margin-top': '20px'}),
    html.Div(id='save-message', style={'margin-top': '10px', 'color': 'green'}),

    # Store to hold user-defined regions data
    #dcc.Store(id='user-defined-regions-store', data=[]),
    # Store to hold how many shapes have been processed into regions
    #dcc.Store(id='processed-shapes-count', data=0),
], style={'width': '80%', 'margin': '0 auto'})


def get_slice_indices(view, slice_idx):
    if view == 'axial':
        return (slice_idx, slice(None), slice(None)), ('z','y','x')
    elif view == 'coronal':
        return (slice(None), slice_idx, slice(None)), ('z','y','x')
    elif view == 'sagittal':
        return (slice(None), slice(None), slice_idx), ('z','y','x')
    else:
        return (slice_idx, slice(None), slice(None)), ('z','y','x')

def transform_coords_to_volume(view, x_range, y_range, slice_idx):
    if view == 'axial':
        return {
            'z_min': slice_idx,
            'z_max': slice_idx,
            'y_min': int(min(y_range)),
            'y_max': int(max(y_range)),
            'x_min': int(min(x_range)),
            'x_max': int(max(x_range))
        }
    elif view == 'coronal':
        return {
            'z_min': int(min(y_range)),
            'z_max': int(max(y_range)),
            'y_min': slice_idx,
            'y_max': slice_idx,
            'x_min': int(min(x_range)),
            'x_max': int(max(x_range))
        }
    elif view == 'sagittal':
        return {
            'z_min': int(min(y_range)),
            'z_max': int(max(y_range)),
            'y_min': int(min(x_range)),
            'y_max': int(max(x_range)),
            'x_min': slice_idx,
            'x_max': slice_idx
        }

def generate_region_inputs(regions):
    inputs = []
    print("CREATING INPUTS")
    for i, r in enumerate(regions):
        print(f"Region {i+1} Threshold: {r['threshold']}")

        # check if the region threshold has been changed


        inputs.append(html.Div([
            html.Label(f"Region {i+1} Threshold"),
            dcc.Input(
                id={'type': 'region-threshold-input', 'index': r['id']},
                type='number',
                min=-1000, max=0, step=10,
                value=r['threshold'],
            )
        ], style={
            'margin-top': '10px',
            'color': f"rgb({int(r['color'][0]*255)}, {int(r['color'][1]*255)}, {int(r['color'][2]*255)})"
        }))
    return inputs

@app.callback(
    Output('slice-index', 'max'),
    Output('slice-index', 'marks'),
    Output('slice-index', 'value'),
    Input('view-selection', 'value'),
    Input('scan-selection', 'value'),
    State('slice-index', 'value')
)
@log_callback
def update_slice_index(view, scan_id, current_slice_idx):
    load_scan_data(scan_id)
    if view == 'axial':
        max_idx = z_dim - 1
    elif view == 'coronal':
        max_idx = y_dim - 1
    elif view == 'sagittal':
        max_idx = x_dim - 1
    else:
        max_idx = z_dim - 1

    if current_slice_idx > max_idx:
        current_slice_idx = max_idx

    step = max(1, max_idx // 10) if max_idx >= 10 else 1
    marks = {i: str(i) for i in range(0, max_idx + 1, step)}
    return max_idx, marks, current_slice_idx



'''@app.callback(
        Output('user-defined-regions-store', 'data'),
        Input({'type': 'region-threshold-input', 'index': ALL}, 'value'),
)
def update_region_thresholds(region_thresholds):
    #regions = json.loads(dash.callback_context.states['user-defined-regions-store']['data'])
    for i, r in enumerate(regions):
        r['threshold'] = region_thresholds[i]
    return regions'''

@app.callback(
    Output('ct-image', 'figure'),
    [
        Input('threshold-lul', 'value'),
        Input('threshold-lll', 'value'),
        Input('threshold-rul', 'value'),
        Input('threshold-rml', 'value'),
        Input('threshold-rll', 'value'),
        Input('slice-index', 'value'),
        Input('view-selection', 'value'),
        Input('scan-selection', 'value'),
        #Input('user-defined-regions-store', 'data'),  # Now an Input instead of State
        #State('enable-drawing', 'value')
    ]
)
@log_callback
def update_image(th_lul, th_lll, th_rul, th_rml, th_rll, slice_idx, view, scan_id):
    load_scan_data(scan_id)
    thresholds = {1: th_lul, 2: th_lll, 3: th_rul, 4: th_rml, 5: th_rll}
        

    # Create modified mask
    modified_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    
        
    # Apply lobe thresholds
    for lobe_label, threshold in thresholds.items():
        if threshold is None:
            continue
        lobe_mask = (lung_mask == lobe_label)
        lobe_ct = ct_volume.copy()
        lobe_ct[~lobe_mask] = -2000
        lobe_mask &= (lobe_ct <= threshold)
        modified_mask[lobe_mask] = lobe_label

    global modified_mask_global
    modified_mask_global = modified_mask.copy()

    slice_indices, axis_order = get_slice_indices(view, slice_idx)
    ct_slice = ct_volume[slice_indices]
    mask_slice = modified_mask[slice_indices]

    if view == 'coronal' or view == 'sagittal':
        ct_slice = np.rot90(np.transpose(ct_slice, (1,0)), k=1)
        mask_slice = np.rot90(np.transpose(mask_slice, (1,0)), k=1)

    ct_image = np.clip((ct_slice + 1000) / 2000, 0, 1)
    overlay = np.zeros(ct_image.shape + (3,), dtype=np.float32)

    # Colors for lobes
    colors = {
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1],
        4: [1, 1, 0],
        5: [1, 0, 1],
    }



    for lbl, clr in colors.items():
        overlay[mask_slice == lbl] = clr

    alpha = 0.3
    combined_image = (1 - alpha) * np.stack([ct_image]*3, axis=-1) + alpha * overlay

    fig = px.imshow(combined_image)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title=f'Scan: {scan_id}, View: {view.capitalize()}, Slice Index: {slice_idx}',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=800,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(dragmode='zoom')

    fig.update_layout(
        dragmode='zoom',
        shapes=[]
    )

    return fig

@app.callback(
    Output('save-message', 'children'),
    Input('save-button', 'n_clicks'),
    State('scan-selection', 'value'),
)
@log_callback
def save_mask(n_clicks, scan_id):
    global dicom_IDs
    print("dicom_IDs: ", dicom_IDs)
    print("scan_id: ", scan_id)
    print(int(scan_id) in dicom_IDs)
    if int(scan_id) in dicom_IDs:
        # save file as nifti file
        save_dir = '../CTA_Nathan'
        print(os.path.exists(save_dir))
        if os.path.exists(save_dir):
            # save ct_volume as nifti file
            # convert to itk image
            itk_image = sitk.GetImageFromArray(ct_volume)
            # copy information from the original image
            itk_image.SetOrigin(mask_itk.GetOrigin())
            itk_image.SetSpacing(mask_itk.GetSpacing())
            itk_image.SetDirection(mask_itk.GetDirection())
            # save the image
            sitk.WriteImage(itk_image, os.path.join(save_dir, f'{scan_id}.nii.gz'))
            print('Image saved successfully.')


    if n_clicks > 0:
        if modified_mask_global is not None and scan_id is not None:
            output_dir = '../thresholded_masks'
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except Exception as e:
                    return f'Error creating directory: {e}'

            # Convert to binary mask (1 for inside, 0 outside)
            binary_mask = np.where(modified_mask_global > 0, 1, 0).astype(np.uint8)
            binary_mask_itk = sitk.GetImageFromArray(binary_mask)
            binary_mask_itk.CopyInformation(mask_itk)

            output_filename = f"{scan_id}_mask_thresholded.nii.gz"
            output_mask_path = os.path.join(output_dir, output_filename)

            try:
                sitk.WriteImage(binary_mask_itk, output_mask_path)
                return f'Mask saved as {output_mask_path}'
            except Exception as e:
                return f'Error saving mask: {e}'
        else:
            return 'Modified mask not available.'
    else:
        return ''
    

if __name__ == '__main__':
    # Pre-load the first scan
    load_scan_data(available_scans[0])
    app.run_server(debug=True)