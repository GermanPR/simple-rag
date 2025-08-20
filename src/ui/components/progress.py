"""Progress indicators and status display components."""

import streamlit as st
from typing import Iterator
from contextlib import contextmanager


class ProgressTracker:
    """Manages progress tracking for multi-step operations."""
    
    def __init__(self, container=None):
        self.container = container or st
        self.current_step = 0
        self.total_steps = 0
        
    def set_total_steps(self, total: int):
        """Set the total number of steps."""
        self.total_steps = total
        self.current_step = 0
    
    def update_step(self, step_name: str, progress: float = None):
        """Update current step with optional progress value."""
        self.current_step += 1
        if progress is None:
            progress = self.current_step / self.total_steps if self.total_steps > 0 else 0.0
        
        # Update progress bar if available
        if hasattr(self, '_progress_bar'):
            self._progress_bar.progress(progress)
        
        # Update status text
        if hasattr(self, '_status_text'):
            self._status_text.text(f"Step {self.current_step}/{self.total_steps}: {step_name}")


@contextmanager
def progress_status(label: str, steps: list[str] = None):
    """Context manager for showing progress with status updates."""
    status_container = st.status(label, expanded=True)
    
    try:
        if steps:
            progress_bar = status_container.progress(0)
            total_steps = len(steps)
            
            def update_progress(step_idx: int, step_name: str):
                progress = (step_idx + 1) / total_steps
                progress_bar.progress(progress)
                status_container.write(f"✅ {step_name}")
            
            yield update_progress
        else:
            yield lambda step_idx, step_name: status_container.write(f"✅ {step_name}")
        
        # Mark as complete
        status_container.update(label=f"✅ {label}", state="complete")
        
    except Exception as e:
        # Mark as error
        status_container.update(label=f"❌ {label}", state="error")
        status_container.error(f"Error: {str(e)}")
        raise


@contextmanager
def upload_progress(total_files: int):
    """Progress tracker specifically for file upload operations."""
    with st.status(f"Processing {total_files} file(s)...", expanded=True) as status:
        progress_bar = st.progress(0)
        processed = 0
        
        def update_file_progress(filename: str, step: str):
            nonlocal processed
            status.write(f"📄 {filename}: {step}")
            
        def complete_file():
            nonlocal processed
            processed += 1
            progress = processed / total_files
            progress_bar.progress(progress)
            
        def set_error(error_msg: str):
            status.update(label="❌ Upload failed", state="error")
            status.error(error_msg)
        
        yield update_file_progress, complete_file, set_error
        
        # Mark as complete
        status.update(label=f"✅ Processed {total_files} file(s)", state="complete")


@contextmanager
def query_progress():
    """Progress tracker for query processing."""
    with st.status("Processing query...", expanded=False) as status:
        steps = []
        
        def add_step(step_name: str, icon: str = "🔄"):
            steps.append(f"{icon} {step_name}")
            status.write(f"{icon} {step_name}")
        
        def complete_step(step_idx: int, icon: str = "✅"):
            if step_idx < len(steps):
                steps[step_idx] = steps[step_idx].replace("🔄", icon)
        
        def set_complete():
            status.update(label="✅ Query completed", state="complete")
            
        def set_error(error_msg: str):
            status.update(label="❌ Query failed", state="error")
            status.error(error_msg)
        
        yield add_step, complete_step, set_complete, set_error


def show_file_processing_steps(filename: str, steps_completed: dict):
    """Show individual file processing steps."""
    cols = st.columns(4)
    step_names = ["Extract", "Chunk", "Embed", "Index"]
    step_icons = ["📄", "✂️", "🧮", "📇"]
    
    for i, (name, icon) in enumerate(zip(step_names, step_icons)):
        with cols[i]:
            if steps_completed.get(name.lower(), False):
                st.success(f"{icon} {name}")
            else:
                st.info(f"⏳ {name}")


def show_query_steps(current_step: str = None):
    """Show query processing steps with current step highlighted."""
    steps = [
        ("analyze", "🔍", "Analyze"),
        ("search", "🔎", "Search"),
        ("generate", "🤖", "Generate"),
        ("complete", "✅", "Complete"),
    ]
    
    cols = st.columns(len(steps))
    for i, (step_key, icon, name) in enumerate(steps):
        with cols[i]:
            if step_key == current_step:
                st.info(f"🔄 {name}")
            elif current_step and i < next((j for j, (key, _, _) in enumerate(steps) if key == current_step), len(steps)):
                st.success(f"{icon} {name}")
            else:
                st.empty()