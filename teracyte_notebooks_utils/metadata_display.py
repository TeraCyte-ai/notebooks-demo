import pandas as pd
from IPython.display import display, HTML, Markdown, clear_output

from .sample import Sample
from .api_utils import get_experiment_metadata, get_sample_metadata, get_user_samples

def display_metadata(exp_metadata: dict, sample_metadata: dict):
    """
    Display selected metadata fields in a single card-style UI, with channels table at the bottom inside the same box.
    - Holder Type
    - Magnification (with 'x')
    - Well Radius (diameter/2, with μm)
    - Number of Timepoints
    - Channels (table, inside the same card)
    """
    from IPython.display import display, HTML
    holder_type = exp_metadata.get('holderType', sample_metadata.get('holderType', 'N/A'))
    magnification = exp_metadata.get('magnification', sample_metadata.get('magnification', 'N/A'))
    magnification_str = f"{magnification}x" if magnification != 'N/A' else 'N/A'
    chip_diameter = sample_metadata.get('diameter', 'N/A')
    try:
        well_radius = float(chip_diameter) / 2
        well_radius_str = f"{well_radius:.2f} μm"
    except Exception:
        well_radius_str = f"{chip_diameter} μm"
    channels = exp_metadata.get('channels', {})
    sequences = sample_metadata.get('sequences', [])
    num_timepoints = len(sequences) if isinstance(sequences, list) else 'N/A'

    # Prepare channel table HTML
    channel_table_html = ''
    if isinstance(channels, dict) and channels:
        # Show Name (channelName), Exposure, Intensity, with capitalized headers
        channel_table_html += '<table style="border-collapse:collapse; min-width:420px; font-size:0.97em; color:#222;">'
        channel_table_html += '<thead><tr>'
        channel_table_html += '<th style="border:1px solid #b3c2cc; padding:6px 10px; background:#e3eaf2; color:#222;">Index</th>'
        channel_table_html += '<th style="border:1px solid #b3c2cc; padding:6px 10px; background:#e3eaf2; color:#222;">Name</th>'
        channel_table_html += '<th style="border:1px solid #b3c2cc; padding:6px 10px; background:#e3eaf2; color:#222;">Exposure</th>'
        channel_table_html += '<th style="border:1px solid #b3c2cc; padding:6px 10px; background:#e3eaf2; color:#222;">Intensity</th>'
        channel_table_html += '</tr></thead><tbody>'
        for idx, ch in channels.items():
            channel_table_html += '<tr>'
            channel_table_html += f'<td style="border:1px solid #b3c2cc; padding:6px 10px; color:#222;">{idx}</td>'
            channel_table_html += f'<td style="border:1px solid #b3c2cc; padding:6px 10px; color:#222;">{ch.get("channelName", "")}</td>'
            channel_table_html += f'<td style="border:1px solid #b3c2cc; padding:6px 10px; color:#222;">{ch.get("exposure", "")}</td>'
            channel_table_html += f'<td style="border:1px solid #b3c2cc; padding:6px 10px; color:#222;">{ch.get("intensity", "")}</td>'
            channel_table_html += '</tr>'
        channel_table_html += '</tbody></table>'
    else:
        channel_table_html = f'<span style="color:#888;">No channel data available</span>'

    # Card-style HTML with icons and flex layout, channels as table
    html = f'''
    <style>
    .tc-meta-card {{
        border: 1px solid #b3c2cc;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8fafc 10%, #e3eaf2 100%);
        padding: 24px 32px;
        margin: 18px 0;
        box-shadow: 0 2px 8px rgba(44,62,80,0.07);
        display: flex;
        flex-direction: column;
        gap: 0px;
        align-items: flex-start;
        font-family: 'Segoe UI', Arial, sans-serif;
        max-width: 1000px;
    }}
    .tc-meta-row {{
        display: flex;
        flex-direction: row;
        gap: 36px;
        align-items: flex-start;
        width: 100%;
    }}
    .tc-meta-item {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        min-width: 120px;
        gap: 6px;
    }}
    .tc-meta-label {{
        font-size: 1.05em;
        color: #2a3a4a;
        font-weight: 700;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .tc-meta-value {{
        font-size: 0.92em;
        color: #1a2a3a;
        font-weight: 600;
    }}
    .tc-meta-channels-table {{
        margin-top: 8px;
        margin-bottom: 2px;
    }}
    </style>
    <div class="tc-meta-card">
        <div class="tc-meta-row">
            <div class="tc-meta-item">
                <span class="tc-meta-label">🔲 Holder Type</span>
                <span class="tc-meta-value">{holder_type}</span>
            </div>
            <div class="tc-meta-item">
                <span class="tc-meta-label">🔬 Magnification</span>
                <span class="tc-meta-value">{magnification_str}</span>
            </div>
            <div class="tc-meta-item">
                <span class="tc-meta-label">🧫 Well Radius</span>
                <span class="tc-meta-value">{well_radius_str}</span>
            </div>
            <div class="tc-meta-item">
                <span class="tc-meta-label">⏱️ Number of Timepoints</span>
                <span class="tc-meta-value">{num_timepoints}</span>
            </div>
        </div>
        <div class="tc-meta-item" style="margin-top:32px; width:100%;">
            <span class="tc-meta-label">🧬 Channels</span>
            <div class="tc-meta-channels-table">{channel_table_html}</div>
        </div>
    </div>
    '''
    display(HTML(html))

def display_hardware_metadata(exp_metadata: dict, sample_metadata: dict):
    # Microscope metadata
    microscope_keys = [
        'deviceID', 'imagerVersion', 'holderType', 'microscope', 'lightSource', 'camera',
        'incubator', 'magnification', 'pixelSize', 'binning', 'width', 'height'
    ]
    microscope_data = {
        k: exp_metadata.get(k, sample_metadata.get(k))
        for k in microscope_keys if k in exp_metadata or k in sample_metadata
    }
    microscope_df = pd.DataFrame(microscope_data.items(), columns=["Key", "Value"])

    # Chip metadata
    chip_keys = [
        'chipType', 'chipID', 'sizeX', 'sizeY', 'blockSizeX', 'blockSizeY',
        'diameter', 'pitch', 'theta', 'bitSize', 'bitMatch'
    ]
    chip_data = {k: sample_metadata[k] for k in chip_keys if k in sample_metadata}
    chip_df = pd.DataFrame(chip_data.items(), columns=["Key", "Value"])

    # Channel metadata
    channels_df = pd.DataFrame.from_dict(exp_metadata.get('channels', {}), orient='index').reset_index()
    channels_df.columns = ['channelIndex'] + list(channels_df.columns[1:])

    # Formatter
    def style_df(df, title):
        return f"""
        <div style="flex: 1; margin-right: 10px;">
            <h4>{title}</h4>
            {df.to_html(index=False, border=1)}
        </div>
        """

    # Build and display
    html_hardware_metadata = f"""
    <h3>Hardware Metadata</h3>
    <div style="display: flex; justify-content: flex-start;">
        {style_df(microscope_df, "Microscope")}
        {style_df(chip_df, "Chip")}
        {style_df(channels_df, "Channels")}
    </div>
    """
    display(HTML(html_hardware_metadata))


def display_sample_metadata(exp_metadata: dict, sample_metadata: dict):
    # Create a mapping of keys to display names for consistent formatting
    key_display_mapping = {
        'name': 'Name',
        'index': 'Index', 
        'id': 'ID',
        'timestamp': 'Timestamp'
    }
    
    sample_data = {}
    for key in ['name', 'index', 'id', 'timestamp']:
        if key in sample_metadata:
            display_name = key_display_mapping[key]
            sample_data[display_name] = sample_metadata[key]
    
    # Add number of timepoints instead of the sequences list
    sequences = sample_metadata.get('sequences', [])
    if isinstance(sequences, list):
        sample_data['Number of Timepoints'] = len(sequences)
    else:
        sample_data['Number of Timepoints'] = 'N/A'
    
    sample_df = pd.DataFrame(sample_data.items(), columns=["Key", "Value"])

    sample_name = sample_metadata.get('name', 'Unnamed Sample')
    display(Markdown(f"### Sample Metadata — `{sample_name}`"))
    display(sample_df.style.hide(axis="index"))



def display_serial_number_records(serial_number, user_id, style='html'):
    """
    Display metadata and records for a given serial number.
    Validates that the serial number exists in the user's samples.
    
    Args:
        serial_number (str): The serial number to query
        user_id (str): The user ID to validate samples against (required)
        style (str): Display style, either 'markdown' (card-style) or 'html' (box-style). Default is 'html'.
    """
    # Clear any existing output
    clear_output(wait=True)
        
    try:
        # Get user's samples to validate serial number
        user_samples = get_user_samples(user_id)
        
        if not user_samples:
            display(HTML(f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="color: #856404; margin: 0;"><strong>Warning:</strong> No samples found for user ID: {user_id}</p>
            </div>
            """))
            return
        
        # Find all samples matching the serial number (sample_id)
        matching_samples = []
        
        for sample in user_samples:
            if isinstance(sample, dict) and sample.get('sample_id') == serial_number:
                matching_samples.append(sample)
        
        if not matching_samples:
            display(HTML(f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="color: #856404; margin: 0;"><strong>Warning:</strong> Serial number '{serial_number}' not found in user's samples. Please verify the serial number.</p>
            </div>
            """))
            return
        
        # Build records with metadata
        records = []
        for sample in matching_samples:
            serial_number_id = sample.get('sample_id')
            assay_id = sample.get('assay_id')
            exp_id = sample.get('exp_id')
            
            # Get metadata
            exp_metadata = get_experiment_metadata(exp_id)
            assay_metadata = get_sample_metadata(serial_number_id, exp_id)
            
            assay_name = assay_metadata.get('name', sample.get('sample_name', 'N/A'))
            experiment_name = exp_metadata.get('exp_name', 'N/A')
            assay_timestamp = assay_metadata.get('timestamp', 'N/A')
            
            # Format timestamp with UTC notation in a nice format
            if assay_timestamp != 'N/A':
                try:
                    # Parse the ISO timestamp and format it nicely
                    from datetime import datetime
                    dt = datetime.fromisoformat(assay_timestamp.replace('Z', '+00:00'))
                    assay_timestamp_display = f"{dt.strftime('%Y-%m-%d %H:%M')} (UTC)"
                except (ValueError, AttributeError):
                    # Fallback to original format if parsing fails
                    assay_timestamp_display = f"{assay_timestamp} (UTC)"
            else:
                assay_timestamp_display = 'N/A'
            
            records.append({
                'Experiment Name': experiment_name,
                'Assay Name': assay_name,
                'Experiment ID': exp_id,
                'Assay ID': assay_id,
                'Assay Timestamp': assay_timestamp_display,
            })
        
        if not records:
            display(Markdown('**No records to display.**'))
            return

        if style == 'markdown':
            md = []
            for i, rec in enumerate(records, 1):
                md.append(f"### Sample - {rec.get('Assay Name', 'N/A')}")
                for k, v in rec.items():
                    md.append(f"- **{k}**: {v}")
                md.append('---')
            display(Markdown('\n'.join(md)))
        elif style == 'html':
            html = [
                '<style>',
                '.tc-card-container { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 12px; }',
                '.tc-card { border: 1px solid #b3c2cc; padding: 18px 20px; border-radius: 12px; min-width: 260px; background: linear-gradient(135deg, #f8fafc 10%, #e3eaf2 100%); }',
                '.tc-card:hover { }',
                '.tc-card-header { font-size: 1.15em; font-weight: 600; color: #2a3b4c; margin-bottom: 10px; letter-spacing: 0.5px; }',
                '.tc-card-list { list-style: none; padding-left: 0; margin: 0; }',
                '.tc-card-list li { margin-bottom: 6px; font-size: 1em; color: #3a4a5c; }',
                '.tc-card-list b { color: #1a2a3a; }',
                '</style>',
                '<div class="tc-card-container">'
            ]
            for i, rec in enumerate(records, 1):
                html.append('<div class="tc-card">')
                html.append(f'<div class="tc-card-header">Sample — {rec.get("Assay Name", "N/A")}</div>')
                html.append('<ul class="tc-card-list">')
                for k, v in rec.items():
                    html.append(f'<li><b>{k}</b>: {v}</li>')
                html.append('</ul></div>')
            html.append('</div>')
            display(HTML(''.join(html)))
        else:
            raise ValueError("Unsupported style. Use 'markdown' or 'html'.")
                
    except Exception as e:
        display(HTML(f"""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <p style="color: #856404; margin: 0;"><strong>Warning:</strong> Error retrieving user samples: {str(e)}</p>
        </div>
        """))