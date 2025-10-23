"""
Wells Groups Manager Module for TeraCyte Data Overview.

This module provides a comprehensive UI for managing wells groups, including:
- Viewing existing groups
- Creating new groups
- Deleting existing groups

The manager handles async API operations and provides user-friendly feedback.
"""

import ipywidgets as widgets
from IPython.display import clear_output
import traceback


def create_wells_groups_manager(sample):
    """
    Create a comprehensive wells groups management UI with three tabs:
    1. View Groups - Display all groups with dropdown selection
    2. Add Group - Create new groups
    3. Remove Group - Delete existing groups
    
    Args:
        sample: Sample object with wells groups functionality
        
    Returns:
        widgets.VBox: Complete wells groups management interface
    """
    
    # Create output areas for each tab
    view_output = widgets.Output()
    add_output = widgets.Output()
    remove_output = widgets.Output()
    
    # Helper functions
    def filter_valid_groups(groups):
        """Filter out empty groups and return only valid ones."""
        if not groups:
            return []
        valid_groups = []
        for group in groups:
            name = group.get('query_name', '').strip()
            indexes = group.get('global_indexes', [])
            # Include groups that have a meaningful name or wells
            if name and name != '(empty query)' or len(indexes) > 0:
                valid_groups.append(group)
        return valid_groups
    
    def format_group_option(group):
        """Format a group for dropdown display."""
        name = group.get('query_name', '').strip()
        if not name or name == '(empty query)':
            name = f"Group {group.get('query_index', 'Unknown')}"
        wells_count = len(group.get('global_indexes', []))
        return f"{name} ({wells_count} wells)"
    
    # ===== TAB 1: VIEW GROUPS =====
    def create_view_groups_tab():
        """Create the view groups tab content."""
        
        # Group selection dropdown
        groups_dropdown = widgets.Dropdown(
            options=[],
            description='Select Group:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        # Info display area
        info_display = widgets.HTML(
            value="<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;'>Select a group to view details</div>",
            layout=widgets.Layout(width='600px', margin='10px 0')
        )
        
        # Refresh button
        refresh_button = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            layout=widgets.Layout(width='120px', height='35px')
        )
        
        # Show all/valid toggle
        show_all_toggle = widgets.Checkbox(
            value=False,
            description='Show empty groups',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='5px 0')
        )
        
        def refresh_groups_list():
            """Refresh the groups dropdown with current data."""
            try:
                all_groups = sample.wells_groups or []
                
                # Filter groups based on toggle
                if show_all_toggle.value:
                    groups = all_groups
                else:
                    groups = filter_valid_groups(all_groups)
                
                if groups:
                    options = [(format_group_option(group), group.get('query_index', i)) 
                              for i, group in enumerate(groups)]
                    groups_dropdown.options = options
                    if not groups_dropdown.value and options:
                        groups_dropdown.value = options[0][1]
                else:
                    if show_all_toggle.value:
                        groups_dropdown.options = [("No groups available", None)]
                    else:
                        groups_dropdown.options = [("No valid groups (try 'Show empty groups')", None)]
                    groups_dropdown.value = None
                update_info_display()
            except Exception as e:
                info_display.value = f"<div style='color: red; padding: 10px;'>Error loading groups: {str(e)}</div>"
        
        def update_info_display():
            """Update the info display based on selected group."""
            selected_index = groups_dropdown.value
            if selected_index is None:
                info_display.value = "<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;'>No group selected</div>"
                return
            
            try:
                all_groups = sample.wells_groups or []
                groups = all_groups if show_all_toggle.value else filter_valid_groups(all_groups)
                
                # Find the group by query_index
                group = None
                for g in groups:
                    if g.get('query_index') == selected_index:
                        group = g
                        break
                
                if group:
                    global_indexes = group.get('global_indexes', [])
                    description = group.get('query_description', '').strip()
                    if not description:
                        description = 'No description'
                    query_index = group.get('query_index', 'N/A')
                    name = group.get('query_name', '').strip()
                    if not name or name == '(empty query)':
                        name = f"Group {query_index}"
                    
                    # Create a preview of indexes (show first 15)
                    if len(global_indexes) == 0:
                        indexes_preview = "No wells selected"
                        preview_color = "#ffeaa7"
                    elif len(global_indexes) <= 15:
                        indexes_preview = ', '.join(map(str, global_indexes))
                        preview_color = "#dff0d8"
                    else:
                        indexes_preview = ', '.join(map(str, global_indexes[:15])) + f', ... (+{len(global_indexes)-15} more)'
                        preview_color = "#dff0d8"
                    
                    # Determine the display color based on group status
                    if len(global_indexes) == 0 and (not name or name.startswith('Group ')):
                        background_color = "#fff3cd"
                        border_color = "#ffeaa7"
                        text_color = "#856404"
                        status_text = "⚠️ Empty Group"
                    else:
                        background_color = "#e7f3ff"
                        border_color = "#bee5eb"
                        text_color = "#0c5460"
                        status_text = "✅ Valid Group"
                    
                    info_html = f"""
                    <div style='padding: 15px; background: {background_color}; border-radius: 5px; border: 1px solid {border_color};'>
                        <h4 style='margin-top: 0; color: {text_color};'>{name} {status_text}</h4>
                        <p><strong>Description:</strong> {description}</p>
                        <p><strong>Query Index:</strong> {query_index}</p>
                        <p><strong>Wells Count:</strong> {len(global_indexes)}</p>
                        <p><strong>Global Indexes:</strong><br>
                        <span style='font-family: monospace; font-size: 12px; background: {preview_color}; padding: 5px; border-radius: 3px; display: inline-block; margin-top: 5px; max-width: 500px; word-wrap: break-word;'>{indexes_preview}</span></p>
                    </div>
                    """
                    info_display.value = info_html
                else:
                    info_display.value = f"<div style='color: red; padding: 10px;'>Group with index '{selected_index}' not found</div>"
            except Exception as e:
                info_display.value = f"<div style='color: red; padding: 10px;'>Error displaying group: {str(e)}</div>"
        
        # Event handlers
        def on_group_change(change):
            update_info_display()
        
        def on_refresh_click(b):
            with view_output:
                clear_output(wait=True)
                print("🔄 Refreshing groups list...")
            refresh_groups_list()
            with view_output:
                print("✅ Groups list refreshed")
        
        def on_toggle_change(change):
            refresh_groups_list()
            with view_output:
                if change['new']:
                    print("ℹ️ Now showing all groups (including empty ones)")
                else:
                    print("ℹ️ Now showing only valid groups")
        
        groups_dropdown.observe(on_group_change, names='value')
        refresh_button.on_click(on_refresh_click)
        show_all_toggle.observe(on_toggle_change, names='value')
        
        # Initial load
        refresh_groups_list()
        
        # Layout
        controls = widgets.HBox([groups_dropdown, refresh_button], 
                              layout=widgets.Layout(align_items='center', margin='10px 0'))
        
        return widgets.VBox([
            widgets.HTML("<h4>👁️ View Wells Groups</h4>"),
            show_all_toggle,
            controls,
            info_display,
            view_output
        ])
    
    # ===== TAB 2: ADD GROUP =====
    def create_add_group_tab():
        """Create the add group tab content."""
        
        # Input fields
        name_input = widgets.Text(
            placeholder='Enter group name',
            description='Name:',
            layout=widgets.Layout(width='400px')
        )
        
        description_input = widgets.Textarea(
            placeholder='Enter group description',
            description='Description:',
            rows=3,
            layout=widgets.Layout(width='500px')
        )
        
        indexes_input = widgets.Textarea(
            placeholder='Enter global indexes (comma-separated): 100, 200, 300, ...',
            description='Global Indexes:',
            rows=4,
            layout=widgets.Layout(width='500px')
        )
        
        # Add button
        add_button = widgets.Button(
            description='➕ Add Group',
            button_style='success',
            layout=widgets.Layout(width='150px', height='40px')
        )
        
        # Helper text
        helper_text = widgets.HTML("""
        <div style='background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 14px;'>
            <strong>💡 Tips:</strong><br>
            • Group names must be unique among existing valid groups<br>
            • Global indexes should be comma-separated numbers (e.g., 100, 200, 300)<br>
            • Group creation is asynchronous - use Refresh in View tab to see new groups<br>
        </div>
        """)
        
        def validate_inputs():
            """Validate the input fields."""
            name = name_input.value.strip()
            description = description_input.value.strip()
            indexes_text = indexes_input.value.strip()
            
            if not name:
                return False, "Group name is required"
            
            if not indexes_text:
                return False, "Global indexes are required"
            
            # Validate indexes format
            try:
                # Handle various separators and formats
                indexes_text = indexes_text.replace(';', ',').replace(' ', ',').replace('\n', ',')
                indexes = []
                for idx in indexes_text.split(','):
                    idx = idx.strip()
                    if idx:
                        indexes.append(int(idx))
                
                if not indexes:
                    return False, "No valid indexes found"
                
                if len(indexes) != len(set(indexes)):
                    return False, "Duplicate indexes found"
                    
                # Validate index range (assuming reasonable bounds)
                if any(idx < 0 for idx in indexes):
                    return False, "Indexes must be non-negative"
                    
            except ValueError as e:
                return False, f"Invalid index format. Use comma-separated numbers only. Error: {str(e)}"
            
            # Check if group name already exists among valid groups
            existing_groups = filter_valid_groups(sample.wells_groups or [])
            existing_names = []
            for group in existing_groups:
                group_name = group.get('query_name', '').strip()
                if group_name and group_name != '(empty query)':
                    existing_names.append(group_name)
            
            if name in existing_names:
                return False, f"Group name '{name}' already exists"
            
            return True, {"name": name, "description": description, "indexes": sorted(indexes)}
        
        def on_add_click(b):
            """Handle add button click."""
            with add_output:
                clear_output(wait=True)
                print("🔄 Validating inputs...")
                
                is_valid, result = validate_inputs()
                if not is_valid:
                    print(f"❌ Validation error: {result}")
                    return
                
                print(f"✅ Validation passed")
                print(f"📝 Creating group '{result['name']}' with {len(result['indexes'])} wells...")
                
                try:
                    response = sample.create_wells_group(
                        name=result['name'],
                        description=result['description'],
                        global_indexes=result['indexes']
                    )
                    
                    if isinstance(response, dict) and 'task_id' in response:
                        print(f"🚀 Group creation submitted successfully!")
                        print(f"   Task ID: {response['task_id']}")
                        print(f"   Status: {response.get('status', 'Unknown')}")
                        print(f"   Group: '{result['name']}' with {len(result['indexes'])} wells")
                        print(f"   Description: {result['description']}")
                        print(f"\n💡 Note: Group creation is asynchronous and may take a few seconds to appear.")
                        print(f"   Use the 'Refresh' button in the View Groups tab to see the new group.")
                    else:
                        print(f"✅ Group creation response: {response}")
                    
                    # Clear the form
                    name_input.value = ""
                    description_input.value = ""
                    indexes_input.value = ""
                    
                except Exception as e:
                    print(f"❌ Error creating group: {str(e)}")
                    print("Full error details:")
                    traceback.print_exc()
        
        add_button.on_click(on_add_click)
        
        # Layout
        form_fields = widgets.VBox([
            name_input,
            description_input,
            indexes_input,
            add_button
        ], layout=widgets.Layout(margin='10px 0'))
        
        return widgets.VBox([
            widgets.HTML("<h4>➕ Add Wells Group</h4>"),
            helper_text,
            form_fields,
            add_output
        ])
    
    # ===== TAB 3: REMOVE GROUP =====
    def create_remove_group_tab():
        """Create the remove group tab content."""
        
        # Group selection for deletion
        delete_dropdown = widgets.Dropdown(
            options=[],
            description='Select Group:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Confirmation checkbox
        confirm_checkbox = widgets.Checkbox(
            value=False,
            description='I confirm I want to delete this group',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0')
        )
        
        # Delete button
        delete_button = widgets.Button(
            description='🗑️ Delete Group',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='40px'),
            disabled=True
        )
        
        # Refresh button
        refresh_delete_button = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            layout=widgets.Layout(width='120px', height='35px')
        )
        
        # Group info display
        delete_info_display = widgets.HTML(
            value="<div style='padding: 10px; background: #fff3cd; border-radius: 5px;'>Select a group to see deletion details</div>",
            layout=widgets.Layout(width='500px', margin='10px 0')
        )
        
        # Warning text
        warning_text = widgets.HTML("""
        <div style='background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; color: #721c24; font-size: 14px;'>
            <strong>⚠️ Warning:</strong><br>
            • Deleting a wells group is permanent and cannot be undone<br>
            • Deletion is asynchronous - use Refresh to see updated groups list
        </div>
        """)
        
        def refresh_delete_groups():
            """Refresh the delete dropdown with current groups."""
            try:
                all_groups = sample.wells_groups or []
                # Only show groups that have names or wells (valid groups)
                groups = filter_valid_groups(all_groups)
                
                if groups:
                    options = [(format_group_option(group), group.get('query_index', i)) 
                              for i, group in enumerate(groups)]
                    delete_dropdown.options = options
                    if not delete_dropdown.value and options:
                        delete_dropdown.value = options[0][1]
                else:
                    delete_dropdown.options = [("No groups available for deletion", None)]
                    delete_dropdown.value = None
                update_delete_info()
            except Exception as e:
                delete_info_display.value = f"<div style='color: red; padding: 10px;'>Error loading groups: {str(e)}</div>"
        
        def update_delete_info():
            """Update the delete info display."""
            selected_index = delete_dropdown.value
            if selected_index is None:
                delete_info_display.value = "<div style='padding: 10px; background: #fff3cd; border-radius: 5px;'>No group selected</div>"
                return
            
            try:
                all_groups = sample.wells_groups or []
                groups = filter_valid_groups(all_groups)
                
                # Find the group by query_index
                group = None
                for g in groups:
                    if g.get('query_index') == selected_index:
                        group = g
                        break
                
                if group:
                    global_indexes = group.get('global_indexes', [])
                    description = group.get('query_description', '').strip()
                    if not description:
                        description = 'No description'
                    name = group.get('query_name', '').strip()
                    if not name or name == '(empty query)':
                        name = f"Group {group.get('query_index', 'Unknown')}"
                    
                    delete_info_html = f"""
                    <div style='padding: 15px; background: #f8d7da; border-radius: 5px; border: 1px solid #f5c6cb; color: #721c24;'>
                        <h4 style='margin-top: 0;'>⚠️ You are about to delete:</h4>
                        <p><strong>Name:</strong> {name}</p>
                        <p><strong>Description:</strong> {description}</p>
                        <p><strong>Wells Count:</strong> {len(global_indexes)}</p>
                    </div>
                    """
                    delete_info_display.value = delete_info_html
                else:
                    delete_info_display.value = f"<div style='color: red; padding: 10px;'>Group with index '{selected_index}' not found</div>"
            except Exception as e:
                delete_info_display.value = f"<div style='color: red; padding: 10px;'>Error displaying group: {str(e)}</div>"
        
        def update_delete_button_state():
            """Enable/disable delete button based on checkbox and selection."""
            delete_button.disabled = not (confirm_checkbox.value and delete_dropdown.value)
        
        def on_delete_group_change(change):
            update_delete_info()
            update_delete_button_state()
        
        def on_confirm_change(change):
            update_delete_button_state()
        
        def on_refresh_delete_click(b):
            with remove_output:
                clear_output(wait=True)
                print("🔄 Refreshing groups list...")
            refresh_delete_groups()
            with remove_output:
                print("✅ Groups list refreshed")
        
        def on_delete_click(b):
            """Handle delete button click."""
            selected_index = delete_dropdown.value
            if selected_index is None or not confirm_checkbox.value:
                return
            
            with remove_output:
                clear_output(wait=True)
                
                # Find the group to get its name
                try:
                    all_groups = sample.wells_groups or []
                    groups = filter_valid_groups(all_groups)
                    group = None
                    for g in groups:
                        if g.get('query_index') == selected_index:
                            group = g
                            break
                    
                    if not group:
                        print(f"❌ Group with index '{selected_index}' not found")
                        return
                    
                    group_name = group.get('query_name', '').strip()
                    if not group_name or group_name == '(empty query)':
                        group_name = f"Group {selected_index}"
                    
                    print(f"🗑️ Deleting group '{group_name}' (index: {selected_index})...")
                    
                    # Use the query_name from the API, even if it's empty
                    api_name = group.get('query_name', '')
                    response = sample.delete_wells_group(api_name)
                    
                    if isinstance(response, dict) and 'task_id' in response:
                        print(f"🚀 Group deletion submitted successfully!")
                        print(f"   Task ID: {response['task_id']}")
                        print(f"   Status: {response.get('status', 'Unknown')}")
                        print(f"   Deleted group: '{group_name}' (index: {selected_index})")
                        print(f"\n💡 Note: Group deletion is asynchronous and may take a few seconds.")
                        print(f"   Use the 'Refresh' button to see updated groups list.")
                    else:
                        print(f"✅ Deletion response: {response}")
                    
                    # Reset the form
                    confirm_checkbox.value = False
                    refresh_delete_groups()
                    
                except Exception as e:
                    print(f"❌ Error deleting group: {str(e)}")
                    traceback.print_exc()
        
        # Event handlers
        delete_dropdown.observe(on_delete_group_change, names='value')
        confirm_checkbox.observe(on_confirm_change, names='value')
        refresh_delete_button.on_click(on_refresh_delete_click)
        delete_button.on_click(on_delete_click)
        
        # Initial load
        refresh_delete_groups()
        
        # Layout
        delete_controls = widgets.HBox([delete_dropdown, refresh_delete_button], 
                                     layout=widgets.Layout(align_items='center', margin='10px 0'))
        
        delete_form = widgets.VBox([
            delete_controls,
            delete_info_display,
            confirm_checkbox,
            delete_button
        ])
        
        return widgets.VBox([
            widgets.HTML("<h4>🗑️ Remove Wells Group</h4>"),
            warning_text,
            delete_form,
            remove_output
        ])
    
    # ===== MAIN TABS CONTAINER =====
    
    # Create tab contents
    view_tab = create_view_groups_tab()
    add_tab = create_add_group_tab()
    remove_tab = create_remove_group_tab()
    
    # Create main tabs widget
    tabs = widgets.Tab()
    tabs.children = [view_tab, add_tab, remove_tab]
    tabs.set_title(0, "👁️ View Groups")
    tabs.set_title(1, "➕ Add Group")
    tabs.set_title(2, "🗑️ Remove Group")
    
    # Header
    header = widgets.HTML("""
    <div style='background: #e8f4fd; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #bee5eb;'>
        <h3 style='margin: 0; color: #0c5460;'>🏷️ Wells Groups Manager</h3>
        <p style='margin: 5px 0 0 0; color: #495057;'>Organize and manage collections of wells for analysis</p>
    </div>
    """)
    
    return widgets.VBox([header, tabs])


def filter_valid_groups(groups):
    """
    Utility function to filter out empty groups and return only valid ones.
    
    Args:
        groups (list): List of group dictionaries
        
    Returns:
        list: Filtered list of valid groups
    """
    if not groups:
        return []
    valid_groups = []
    for group in groups:
        name = group.get('query_name', '').strip()
        indexes = group.get('global_indexes', [])
        # Include groups that have a meaningful name or wells
        if name and name != '(empty query)' or len(indexes) > 0:
            valid_groups.append(group)
    return valid_groups


def format_group_option(group):
    """
    Utility function to format a group for dropdown display.
    
    Args:
        group (dict): Group dictionary
        
    Returns:
        str: Formatted group display string
    """
    name = group.get('query_name', '').strip()
    if not name or name == '(empty query)':
        name = f"Group {group.get('query_index', 'Unknown')}"
    wells_count = len(group.get('global_indexes', []))
    return f"{name} ({wells_count} wells)"
