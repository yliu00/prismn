import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import sys
import io
from contextlib import redirect_stdout
from layer_calculation_demo import demo_layer_calculations


async def execute_layer_calculation(model_id):
    """Execute the layer calculation demo and capture its output"""
    try:
        # Capture the output from the demo function
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            # Run the existing demo function
            success = await demo_layer_calculations()
        
        # Get the captured output
        output_text = output_buffer.getvalue()
        
        return {
            'success': success,
            'output': output_text
        }
        
    except Exception as e:
        return {
            'success': False,
            'output': f"Error in calculation: {e}"
        }

def show_main_window():
    splash_root.destroy()
    # Your main window code here
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Prism VLLM Layer Calculation Output")
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Handle fullscreen toggle
    def toggle_fullscreen(event=None):
        root.attributes('-fullscreen', not root.attributes('-fullscreen'))
        if root.attributes('-fullscreen'):
            # Center content when going fullscreen
            center_content()
    
    def center_content():
        """Center the content when in fullscreen"""
        if 'canvas' in locals():
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            
            # Center the scrollable frame horizontally
            scrollable_frame.update_idletasks()
            frame_width = scrollable_frame.winfo_reqwidth()
            
            if canvas_width > frame_width:
                x_offset = (canvas_width - frame_width) // 2
                canvas.coords(canvas.find_all()[0], x_offset, 0)
    
    # Bind F11 for fullscreen toggle
    root.bind('<F11>', toggle_fullscreen)
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))

    # Row for label and entry
    top_row = tk.Frame(root, bg='#dec39b')
    top_row.pack(fill="x", pady=10)

    # Center the content
    center_frame = tk.Frame(top_row, bg='#dec39b')
    center_frame.pack(expand=True)

    label = tk.Label(center_frame, text="Choose Your Preference", fg="white", bg="#dec39b", anchor="w", font=("Helvetica", 14))
    label.pack(side="left", padx=10)

    # Create dropdown for preference selection
    preference_var = tk.StringVar(value="Low Latency")
    preference_dropdown = ttk.Combobox(center_frame, textvariable=preference_var, 
                                      values=["Low Latency", "Low Cost"], 
                                      state="readonly", width=15, font=("Helvetica", 11))
    preference_dropdown.pack(side="left", padx=10)

    def clear_output():
        """Clear all labels from the scrollable frame"""
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

    def add_output_line(text, font_size=10, color="black", bold=False):
        """Add a new label to the output with consistent styling"""
        font_weight = "bold" if bold else "normal"
        
        label = tk.Label(scrollable_frame, text=text, 
                        font=("Helvetica", font_size, font_weight),
                        fg=color, bg="#f5f0e8", anchor="w", justify="left")
        label.pack(fill="x", padx=10, pady=2)
        return label

    def run_calculation():
        preference = preference_var.get()
        model_id = "meta-llama/Llama-3.2-1B"  # Default model
        
        # Clear previous results
        clear_output()
        
        # Show loading message
        add_output_line("🚀 Starting Layer Calculation Demo...", font_size=12, bold=True)
        add_output_line(f"📥 Model: {model_id}", font_size=11)
        add_output_line(f"🎯 Preference: {preference}", font_size=11)
        add_output_line("=" * 60, font_size=10)
        add_output_line("")  # Empty line
        
        # Run calculation in a separate thread to avoid blocking UI
        def run_async_calculation():
            try:
                # Run the async calculation
                result = asyncio.run(execute_layer_calculation(model_id))
                
                # Update UI in main thread
                root.after(0, lambda: display_results(result))
            except Exception as e:
                root.after(0, lambda: display_error(str(e)))
        
        # Start calculation in background thread
        calculation_thread = threading.Thread(target=run_async_calculation)
        calculation_thread.daemon = True
        calculation_thread.start()

    def display_results(result):
        if not result or not result.get('success', False):
            add_output_line("❌ Calculation failed!", font_size=12, color="red", bold=True)
            if result and 'output' in result:
                # Split output into lines and display each as a label
                lines = result['output'].split('\n')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        add_output_line(line)
            return
            
        # Clear and show the captured output from the demo
        clear_output()
        lines = result['output'].split('\n')
        for line in lines:
            if line.strip():  # Skip empty lines
                # Style different types of lines consistently
                if line.startswith("🚀") or line.startswith("📥") or line.startswith("✅") or line.startswith("❌"):
                    add_output_line(line, font_size=11, bold=True)
                elif line.startswith("=") or line.startswith("-"):
                    add_output_line(line, font_size=10)
                elif line.startswith("   ") and not line.startswith("      "):
                    add_output_line(line, font_size=10, bold=True)
                else:
                    add_output_line(line, font_size=10)

    def display_error(error_msg):
        add_output_line(f"❌ Error: {error_msg}", font_size=12, color="red", bold=True)

    run_button = ttk.Button(root, text="Run", command=run_calculation)
    run_button.pack(pady=10)

    # Set main window background
    root.configure(bg='#dec39b')
    
    # Create a frame for the scrollable content
    main_frame = tk.Frame(root, bg='#dec39b')
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create a canvas and scrollbar for scrolling
    canvas = tk.Canvas(main_frame, bg="#f5f0e8", highlightthickness=0)  # Lighter hue
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f5f0e8")  # Lighter hue
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Bind mousewheel to canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Update centering function to use the actual canvas
    def center_content():
        """Center the content when in fullscreen"""
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Center the scrollable frame both horizontally and vertically
        scrollable_frame.update_idletasks()
        frame_width = scrollable_frame.winfo_reqwidth()
        frame_height = scrollable_frame.winfo_reqheight()
        
        # Calculate offsets for centering
        x_offset = max(0, (canvas_width - frame_width) // 2)
        y_offset = max(0, (canvas_height - frame_height) // 2)
        
        # Update the position of the scrollable frame
        canvas.coords(canvas.find_all()[0], x_offset, y_offset)
        
        # Also center the main frame content
        main_frame.update_idletasks()
        main_width = main_frame.winfo_width()
        main_height = main_frame.winfo_height()
        
        # Center the main frame within the root window
        root.update_idletasks()
        root_width = root.winfo_width()
        root_height = root.winfo_height()
        
        main_x_offset = max(0, (root_width - main_width) // 2)
        main_y_offset = max(0, (root_height - main_height) // 2)
    
    # Update the toggle_fullscreen function to use the local center_content
    def toggle_fullscreen(event=None):
        root.attributes('-fullscreen', not root.attributes('-fullscreen'))
        if root.attributes('-fullscreen'):
            # Center content when going fullscreen with a slight delay
            root.after(100, center_content)
    
    # Rebind the fullscreen toggle
    root.unbind('<F11>')
    root.bind('<F11>', toggle_fullscreen)
    
    # Bind window resize events to re-center content
    def on_resize(event):
        if root.attributes('-fullscreen'):
            root.after(50, center_content)
    
    root.bind('<Configure>', on_resize)
    
    ttk.Button(root, text="Quit", command=root.destroy).pack(side="right", padx=10, pady=10)
    root.mainloop()

splash_root = tk.Tk()
splash_root.geometry("800x600")
splash_root.title("Loading...")
splash_label = tk.Label(splash_root, text="Welcome to Prism VLLM", fg="white",font=("Helvetica", 28, "bold"), background='#dec39b')
splash_root.configure(bg='#dec39b') 
splash_label.pack(expand=True)

# Show main window after 3 seconds
splash_root.after(3000, show_main_window)
splash_root.mainloop()