# src/app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import SimpleITK as sitk
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Paths to the data
# Adjust these paths based on your directory structure
ct_volume_path = './CTA_Nathan/1056698'  # Path to DICOM series
lung_mask_path = './Lung_Masks/1056698_mask.nii.gz'  # Path to lung mask

# Load the CT volume (DICOM series)
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(ct_volume_path)
reader.SetFileNames(dicom_names)
full_volume_itk = reader.Execute()

# Load the lung mask
mask_itk = sitk.ReadImage(lung_mask_path)
lung_mask = sitk.GetArrayFromImage(mask_itk)

# Convert CT volume to NumPy array
ct_volume = sitk.GetArrayFromImage(full_volume_itk)

# Get dimensions
z_dim, y_dim, x_dim = ct_volume.shape

# Initialize global variable to store the modified mask
# Note: Using a global variable is suitable for single-user, local environments
modified_mask_global = None

# Define the layout of the app
app.layout = html.Div([
    html.H1('Lung Lobe Threshold Adjustment'),

    # Graph to display the image
    dcc.Graph(id='ct-image'),

    # Controls (View selection, Slice index, and Threshold inputs)
    html.Div([
        # View Selection Dropdown
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

        # Slice Index Slider
        html.Div([
            html.Label('Slice Index'),
            dcc.Slider(
                id='slice-index',
                min=0,
                max=z_dim - 1,
                step=1,
                value=z_dim // 2,
                marks={i: str(i) for i in range(0, z_dim, max(1, z_dim // 10))},
                updatemode='drag',  # Update continuously as the slider is dragged
            ),
        ], style={'margin-top': '10px'}),

        # Threshold Inputs for Each Lobe
        html.Div([
            html.Label('Left Upper Lobe Threshold'),
            dcc.Input(
                id='threshold-lul',
                type='number',
                min=-1000,
                max=0,
                step=10,
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
                step=10,
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
                step=10,
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
                step=10,
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
                step=10,
                value=-720,
            ),
        ], style={'margin-top': '10px'}),
    ], style={'margin-top': '20px'}),

    # Save Button
    html.Button('Save Mask', id='save-button', n_clicks=0, style={'margin-top': '20px'}),

    # Message Output
    html.Div(id='save-message', style={'margin-top': '10px', 'color': 'green'}),

], style={'width': '80%', 'margin': '0 auto'})

# Callback to update the slice index slider based on the selected view
@app.callback(
    Output('slice-index', 'max'),
    Output('slice-index', 'marks'),
    Output('slice-index', 'value'),
    Input('view-selection', 'value'),
    State('slice-index', 'value')
)
def update_slice_index(view, current_slice_idx):
    if view == 'axial':
        max_idx = z_dim - 1
    elif view == 'coronal':
        max_idx = y_dim - 1
    elif view == 'sagittal':
        max_idx = x_dim - 1
    else:
        max_idx = z_dim - 1  # Default to axial if view is not recognized

    # Adjust the current slice index if it's out of bounds
    if current_slice_idx > max_idx:
        current_slice_idx = max_idx

    # Update the slider marks
    step = max(1, max_idx // 10)
    marks = {i: str(i) for i in range(0, max_idx + 1, step)}

    return max_idx, marks, current_slice_idx

# Main callback to update the image and store the modified mask
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
    ]
)
def update_image(th_lul, th_lll, th_rul, th_rml, th_rll, slice_idx, view):
    global modified_mask_global  # Access the global variable

    thresholds = {
        1: th_lul,  # Left Upper Lobe
        2: th_lll,  # Left Lower Lobe
        3: th_rul,  # Right Upper Lobe
        4: th_rml,  # Right Middle Lobe
        5: th_rll,  # Right Lower Lobe
    }

    # Create a modified mask based on thresholds
    modified_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    for lobe_label, threshold in thresholds.items():
        if threshold is None:
            continue  # Skip if no threshold value
        lobe_mask = (lung_mask == lobe_label)
        lobe_ct = ct_volume.copy()
        lobe_ct[~lobe_mask] = -2000  # Exclude other lobes
        lobe_mask &= (lobe_ct <= threshold)
        modified_mask[lobe_mask] = lobe_label  # Keep label for visualization

    # Update the global variable
    modified_mask_global = modified_mask.copy()

    # Extract the desired slice based on the selected view
    if view == 'axial':
        ct_slice = ct_volume[slice_idx, :, :]
        mask_slice = modified_mask[slice_idx, :, :]
        # No need to transpose or rotate
    elif view == 'coronal':
        ct_slice = ct_volume[:, slice_idx, :]
        mask_slice = modified_mask[:, slice_idx, :]
        # Transpose and rotate to orient the image correctly
        ct_slice = np.rot90(np.transpose(ct_slice, (1, 0)), k=1)
        mask_slice = np.rot90(np.transpose(mask_slice, (1, 0)), k=1)
    elif view == 'sagittal':
        ct_slice = ct_volume[:, :, slice_idx]
        mask_slice = modified_mask[:, :, slice_idx]
        # Transpose and rotate to orient the image correctly
        ct_slice = np.rot90(np.transpose(ct_slice, (1, 0)), k=1)
        mask_slice = np.rot90(np.transpose(mask_slice, (1, 0)), k=1)
    else:
        # Default to axial view
        ct_slice = ct_volume[slice_idx, :, :]
        mask_slice = modified_mask[slice_idx, :, :]

    # Normalize the CT image for display
    ct_image = np.clip((ct_slice + 1000) / 2000, 0, 1)

    # Prepare the overlay with different colors for each lobe
    colors = {
        1: [1, 0, 0],    # Red
        2: [0, 1, 0],    # Green
        3: [0, 0, 1],    # Blue
        4: [1, 1, 0],    # Yellow
        5: [1, 0, 1],    # Magenta
    }

    overlay = np.zeros(ct_image.shape + (3,), dtype=np.float32)  # RGB image
    for lobe_label, color in colors.items():
        overlay[mask_slice == lobe_label] = color

    # Combine the CT image and the overlay
    alpha = 0.3
    combined_image = (1 - alpha) * np.stack([ct_image]*3, axis=-1) + alpha * overlay

    # Create the figure using Plotly
    fig = px.imshow(combined_image)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f'View: {view.capitalize()}, Slice Index: {slice_idx}',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=800,  # Increase the height of the image
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Enable zooming and panning
    fig.update_layout(
        dragmode='zoom',  # Default drag mode is zoom
    )
    fig.update_layout(
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            constrain='domain',
        )
    )

    return fig

# Callback to handle the save button click
@app.callback(
    Output('save-message', 'children'),
    Input('save-button', 'n_clicks'),
)
def save_mask(n_clicks):
    if n_clicks > 0:
        if modified_mask_global is not None:
            # Define the output directory
            output_dir = '../thresholded_masks'  # Adjust the path if necessary

            # Check if the directory exists; if not, create it
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except Exception as e:
                    return f'Error creating directory: {e}'

            # Convert the modified mask to binary (0 and 1)
            binary_mask = np.where(modified_mask_global > 0, 1, 0).astype(np.uint8)

            # Convert the binary mask to SimpleITK Image
            binary_mask_itk = sitk.GetImageFromArray(binary_mask)

            # Copy spatial metadata from the original mask
            binary_mask_itk.CopyInformation(mask_itk)

            # Generate the output filename
            original_mask_filename = os.path.basename(lung_mask_path)
            filename_wo_ext = os.path.splitext(original_mask_filename)[0]
            # Handle double extensions (e.g., .nii.gz)
            if filename_wo_ext.endswith('.nii'):
                filename_wo_ext = os.path.splitext(filename_wo_ext)[0]
            output_filename = f"{filename_wo_ext}_thresholded.nii.gz"
            output_mask_path = os.path.join(output_dir, output_filename)

            try:
                # Save the binary mask
                sitk.WriteImage(binary_mask_itk, output_mask_path)
                return f'Mask saved as {output_mask_path}'
            except Exception as e:
                return f'Error saving mask: {e}'
        else:
            return 'Modified mask not available.'
    else:
        return ''
    
if __name__ == '__main__':
    app.run_server(debug=True)