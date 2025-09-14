import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import io
import re
from contextlib import redirect_stdout
from layer_calculation_demo import demo_layer_calculations

async def execute_layer_calculation(model_id, preference):
    """Execute the layer calculation demo and capture its output"""
    try:
        # Capture the output from the demo function
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            # Run the existing demo function
            success = await demo_layer_calculations(preference)
        
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
    root.title("Prismn VLLM Layer Router Tool")
    
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

    # Set main window background
    root.configure(bg='#282828')

    # Create a frame for the scrollable content
    main_frame = tk.Frame(root, bg='#282828')
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)

    # Create main content frame with dark background
    content_frame = tk.Frame(main_frame, bg='#282828', relief='solid', bd=2)
    content_frame.pack(expand=True, fill="both")

    # Row for label and entry
    top_row = tk.Frame(content_frame, bg='#282828')
    top_row.pack(fill="x", padx=20, pady=20)

    # Center the content
    center_frame = tk.Frame(top_row, bg='#282828')
    center_frame.pack(expand=True)

    # Create a bordered box for the preference selection
    pref_box = tk.Frame(center_frame, bg='#282828', relief='solid', bd=1, highlightbackground='#ff8c42', highlightthickness=1)
    pref_box.pack(fill='x', padx=10)

    label = tk.Label(pref_box, text="* Choose Your Preference", fg="#ff8c42", bg='#282828', anchor="w", font=("Consolas", 12))
    label.pack(side="left", padx=15, pady=10)

    # Create dropdown for preference selection
    preference_var = tk.StringVar(value="Low Carbon Emissions")
    preference_dropdown = ttk.Combobox(pref_box, textvariable=preference_var, 
                                      values=["Low Latency", "Low Carbon Emissions"], 
                                      state="readonly", width=20, font=("Consolas", 11))
    preference_dropdown.pack(side="right", padx=15, pady=20)

    second_pref_box = tk.Frame(center_frame, bg='#282828', relief='solid', bd=1, highlightbackground='#ff8c42', highlightthickness=1)
    second_pref_box.pack(fill='x', padx=10, pady=(10, 0))

    model_label = tk.Label(second_pref_box, text="* Choose Your Model", fg="#ff8c42", bg='#282828', anchor="w", font=("Consolas", 12))
    model_label.pack(side="left", padx=15, pady=10)

    # Create dropdown for model selection
    model_var = tk.StringVar(value="meta-llama/Llama-3.2-1B")
    model_preference_dropdown = ttk.Combobox(second_pref_box, textvariable=model_var, 
                                      values=["meta-llama/Llama-3.2-1B"], 
                                      state="readonly", width=20, font=("Consolas", 11))
    model_preference_dropdown.pack(side="right", padx=15, pady=10)

    # Helper functions for output and calculation
    def clear_output():
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

    def add_output_line(text, font_size=10, color="#ff8c42", bold=False):
        emoji_pattern = re.compile(r'^[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]')
        if emoji_pattern.match(text):
            color = "#00bfff"  # Or any color you prefer for emoji lines
        font_weight = "bold" if bold else "normal"
        
        # Calculate dynamic wrap length based on window size
        def get_wrap_length():
            try:
                # Get the current canvas width
                canvas_width = canvas.winfo_width()
                if canvas_width > 1:  # Make sure canvas is initialized
                    # Use 80% of canvas width for text wrapping
                    return int(canvas_width * 0.8)
                else:
                    # Fallback to a reasonable default
                    return 800
            except:
                return 800
        
        # Create label with dynamic wrapping
        label = tk.Label(scrollable_frame, text=text, 
                        font=("Consolas", font_size, font_weight), 
                        fg=color, bg="#282828", 
                        anchor="w", justify="left", 
                        wraplength=get_wrap_length())
        label.pack(fill="x", padx=15, pady=3)
        # Update wrapping when window is resized
        def update_wrap(event=None):
            try:
                new_wrap = get_wrap_length()
                label.configure(wraplength=new_wrap)
            except:
                pass
        canvas.bind('<Configure>', update_wrap)
        return label

    def display_results(result):
        if not result or not result.get('success', False):
            add_output_line("❌ Calculation failed!", font_size=12, color="red", bold=True)
            if result and 'output' in result:
                lines = result['output'].split('\n')
                for line in lines:
                    if line.strip():
                        add_output_line(line)
            return
        clear_output()
        lines = result['output'].split('\n')
        for line in lines:
            if line.strip():
                if line.startswith("🚀") or line.startswith("📥") or line.startswith("✅") or line.startswith("❌"):
                    add_output_line(line, font_size=12, bold=True)
                elif line.startswith("=") or line.startswith("-"):
                    add_output_line(line, font_size=12)
                elif line.startswith("   ") and not line.startswith("      "):
                    add_output_line(line, font_size=12, bold=True)
                else:
                    add_output_line(line, font_size=12)

    def display_error(error_msg):
        add_output_line(f"❌ Error: {error_msg}", font_size=12, color="red", bold=True)

    def run_calculation():
        preference = preference_var.get()
        model_id = "meta-llama/Llama-3.2-1B"
        clear_output()
        add_output_line("🚀 Starting Layer Router Demo...", font_size=12, bold=True)
        add_output_line(f"📥 Model: {model_id}", font_size=12)
        add_output_line(f"🎯 Preference: {preference}", font_size=12)
        add_output_line("=" * 60, font_size=10)
        add_output_line("")
        def run_async_calculation():
            try:
                result = asyncio.run(execute_layer_calculation(model_id, preference))
                root.after(0, lambda: display_results(result))
            except Exception as e:
                root.after(0, lambda: display_error(str(e)))
        calculation_thread = threading.Thread(target=run_async_calculation)
        calculation_thread.daemon = True
        calculation_thread.start()

    # Create button row underneath, centered
    button_frame = tk.Frame(center_frame, bg='#282828')
    button_frame.pack(fill='x', padx=10, pady=(10, 0))
    button_center = tk.Frame(button_frame, bg='#282828')
    button_center.pack(expand=True)
    run_button = tk.Button(button_center, text="▶ Run Calculation", command=run_calculation,
                          fg="#ff8c42", bg='#282828', font=("Consolas", 11, "bold"),
                          relief='solid', bd=1, highlightbackground='#ff8c42', highlightthickness=1)
    run_button.pack(side="left", padx=(0, 10), pady=10)
    quit_button = tk.Button(button_center, text="✖ Quit", command=root.destroy,
                           fg="#ff8c42", bg='#282828', font=("Consolas", 11, "bold"),
                           relief='solid', bd=1, highlightbackground='#ff8c42', highlightthickness=1)
    quit_button.pack(side="left", padx=(10, 0), pady=10)

    def clear_output():
        """Clear all labels from the scrollable frame"""
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

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
                    add_output_line(line, font_size=11)
                elif line.startswith("   ") and not line.startswith("      "):
                    add_output_line(line, font_size=11, bold=True)
                else:
                    add_output_line(line, font_size=11)

    def display_error(error_msg):
        add_output_line(f"❌ Error: {error_msg}", font_size=12, color="red", bold=True)

    # Set main window background
    root.configure(bg='#282828')
    
    # Create a frame for the scrollable content
    main_frame = tk.Frame(root, bg='#f5f0e8')
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create main content frame with dark background
    content_frame = tk.Frame(main_frame, bg='#282828', relief='solid', bd=2)
    content_frame.pack(expand=True, fill="both")
    
    # Create a canvas and scrollbar for scrolling
    canvas = tk.Canvas(content_frame, bg="#282828", highlightthickness=0)
    scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#282828")
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack canvas and scrollbar in the content frame
    canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 20))
    scrollbar.pack(side="right", fill="y", padx=(0, 20), pady=(0, 20))
    
    # Bind mousewheel to canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Add initial welcome message
    add_output_line("🚀 Prismn VLLM Layer Router Tool", font_size=14, bold=True)
    add_output_line("=" * 50, font_size=10)
    add_output_line("")
    add_output_line("Select your preference above and click 'Run Calculation' to begin.", font_size=11)
    add_output_line("")
    add_output_line("This tool will analyze your model and optimize layer distribution", font_size=11)
    add_output_line("across available GPU peers for maximum efficiency.", font_size=11)
    
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
        
        # Update text wrapping for all existing labels
        for widget in scrollable_frame.winfo_children():
            if isinstance(widget, tk.Label):
                try:
                    new_wrap = int(canvas_width * 0.8)
                    widget.configure(wraplength=new_wrap)
                except:
                    pass
    
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
    
    root.mainloop()

splash_root = tk.Tk()
splash_root.geometry("900x600")
splash_root.title("Prismn VLLM")
splash_root.configure(bg='#f5f0e8')  # Light beige background

# Create main content frame with dark background
content_frame = tk.Frame(splash_root, bg='#282828', relief='solid', bd=2)
content_frame.place(relx=0.5, rely=0.5, anchor='center', width=800, height=500)

# Top message box
top_frame = tk.Frame(content_frame, bg='#282828')
top_frame.pack(fill='x', padx=20, pady=(30, 20))

top_box = tk.Frame(top_frame, bg='#282828', relief='solid', bd=1, highlightbackground='#ff8c42', highlightthickness=1)
top_box.pack(fill='x')

# Center title with blocky font effect
center_frame = tk.Frame(content_frame, bg='#282828')
center_frame.pack(expand=True, fill='both', padx=20, pady=20)

# Create blocky "WELCOME TO PRISMN" text
title_text = "WELCOME TO PRISMN"
title_label = tk.Label(center_frame, text=title_text, 
                      fg='#ff8c42', bg='#282828', 
                      font=("Consolas", 24, "bold"), 
                      justify='center')
title_label.pack(expand=True)

# Show main window after 2 seconds
splash_root.after(2000, show_main_window)
splash_root.mainloop()