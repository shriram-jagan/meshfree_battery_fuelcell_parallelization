import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, scrolledtext


def load_preset_values(preset_file_path):
    # Initialize variables
    preset_values = {}

    # Read preset values from the text file
    with open(preset_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(": ")
            preset_values[key] = value

    return preset_values


def generate_text_and_execute_script():
    # Retrieve input values
    x_min = x_min_entry.get()
    x_max = x_max_entry.get()
    y_min = y_min_entry.get()
    y_max = y_max_entry.get()
    t = t_entry.get()
    nt = nt_entry.get()
    single_grain = single_grain_entry.get()
    differential_method = differential_method_entry.get()
    integral_method = integral_method_entry.get()
    n_boundaries = n_boundaries_entry.get()
    Tk = Tk_entry.get()
    c_max = c_max_entry.get()
    k_con = k_con_entry.get()
    Dx_div_Dy = Dx_div_Dy_entry.get()
    j_applied = j_applied_entry.get()
    E = E_entry.get()
    nu = nu_entry.get()
    c = c_entry.get()
    ini_charge_state = ini_charge_state_entry.get()
    ini_potential = ini_potential_entry.get()

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Generate a random filename for the text file
    random_filename = "input"  # next(tempfile._get_candidate_names())
    text_file_path = os.path.join(script_directory, random_filename + ".txt")

    # Write input values to the text file
    with open(text_file_path, "w") as file:
        file.write(f"x_min: {x_min}\n")  #
        file.write(f"x_max: {x_max}\n")  #
        file.write(f"y_min: {y_min}\n")  #
        file.write(f"y_max: {y_max}\n")  #
        file.write(f"t: {t}\n")  #
        file.write(f"nt: {nt}\n")  #
        file.write(f"single_grain: {single_grain}\n")  #
        file.write(f"differential_method: {differential_method}\n")  #
        file.write(f"integral_method: {integral_method}\n")  #
        file.write(f"n_boundaries: {n_boundaries}\n")  #
        file.write(f"Tk: {Tk}\n")  #
        file.write(f"c_max: {c_max}\n")  #
        file.write(f"k_con: {k_con}\n")  #
        file.write(f"Dx_div_Dy: {Dx_div_Dy}\n")  #
        file.write(f"j_applied: {j_applied}\n")  #
        file.write(f"E: {E}\n")  #
        file.write(f"nu: {nu}\n")  #
        file.write(f"c: {c}\n")  #
        file.write(f"ini_charge_state: {ini_charge_state}\n")  #
        file.write(f"ini_potential: {ini_potential}\n")  #

    # Inform the user that the text file has been generated
    # result_label.config(text="Text file generated successfully!")

    # Execute the other Python script with the input values as arguments
    script_path = "main.py"
    command = ["python", script_path]

    # Start the subprocess and capture its output
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    # Display the output in the result text widget
    output_text = ""
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            output_text += line.strip() + "\n"
            output_text = output_text[-500:]  # Limiting the output to 500 characters
            output_text_widget.config(state=tk.NORMAL)
            output_text_widget.delete(1.0, tk.END)  # Clear previous output
            output_text_widget.insert(tk.END, output_text)
            output_text_widget.config(state=tk.DISABLED)  # Prevent editing of output
            output_text_widget.see(tk.END)  # Scroll to the bottom

    # Wait for the subprocess to finish
    process.wait()

    # Inform the user that the script has been executed
    # result_label.config(text=result_label.cget("text") + "\nScript executed successfully!")


# Create GUI window
root = tk.Tk()
root.title("MeshfreeDOM")

# Paragraph at the top
intro_text = """
Welcome to the Text File Generator and Script Executor.
Please fill out the following information:
"""
intro_label = tk.Label(root, text=intro_text)
intro_label.grid(row=0, column=0, columnspan=6, padx=2, pady=2)

# Load preset values from a different text file
preset_file_path = "original.txt"  # Replace with the actual path of the preset file
preset_values = load_preset_values(preset_file_path)

# # Display preset values
# preset_label_text = "Preset Values:"
# preset_label = tk.Label(root, text=preset_label_text, font=("Arial", 10, "bold"))
# preset_label.grid(row=1, column=0, padx=2, pady=2, sticky="w")

# geometry:
geometry_label = tk.Label(root, text="Geometry", font=("Arial", 10, "bold"))
geometry_label.grid(row=1, column=0, columnspan=2, padx=2, pady=2)

# cell_thick input with preset value
x_min_label = tk.Label(root, text="x_min:")
x_min_label.grid(row=2, column=0, padx=2, pady=2)
x_min_entry = tk.Entry(root)
x_min_entry.insert(0, preset_values.get("x_min", ""))
x_min_entry.grid(row=2, column=1, padx=2, pady=2)

# n_cell_length input with preset value
x_max_label = tk.Label(root, text="x_max:")
x_max_label.grid(row=3, column=0, padx=2, pady=2)
x_max_entry = tk.Entry(root)
x_max_entry.insert(0, preset_values.get("x_max", ""))
x_max_entry.grid(row=3, column=1, padx=2, pady=2)

# n_cell_width input with preset value
y_min_label = tk.Label(root, text="y_min:")
y_min_label.grid(row=4, column=0, padx=2, pady=2)
y_min_entry = tk.Entry(root)
y_min_entry.insert(0, preset_values.get("y_min", ""))
y_min_entry.grid(row=4, column=1, padx=2, pady=2)

# Additional front_glass_thick input with preset value
y_max_label = tk.Label(root, text="y_max:")
y_max_label.grid(row=5, column=0, padx=2, pady=2)
y_max_entry = tk.Entry(root)
y_max_entry.insert(0, preset_values.get("y_max", ""))
y_max_entry.grid(row=5, column=1, padx=2, pady=2)

# Additional front_glass_thick input with preset value
n_boundaries_label = tk.Label(root, text="n_boundaries:")
n_boundaries_label.grid(row=6, column=0, padx=2, pady=2)
n_boundaries_entry = tk.Entry(root)
n_boundaries_entry.insert(0, preset_values.get("n_boundaries", ""))
n_boundaries_entry.grid(row=6, column=1, padx=2, pady=2)

# geometry:
time_label = tk.Label(root, text="Time Steps", font=("Arial", 10, "bold"))
time_label.grid(row=7, column=0, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
t_label = tk.Label(root, text="t:")
t_label.grid(row=8, column=0, padx=2, pady=2)
t_entry = tk.Entry(root)
t_entry.insert(0, preset_values.get("t", ""))
t_entry.grid(row=8, column=1, padx=2, pady=2)

# Additional back_encap_thick input with preset value
nt_label = tk.Label(root, text="nt:")
nt_label.grid(row=9, column=0, padx=2, pady=2)
nt_entry = tk.Entry(root)
nt_entry.insert(0, preset_values.get("nt", ""))
nt_entry.grid(row=9, column=1, padx=2, pady=2)

# geometry:
singlegrain_label = tk.Label(root, text="Single Grain?", font=("Arial", 10, "bold"))
singlegrain_label.grid(row=10, column=0, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
single_grain_label = tk.Label(root, text="single_grain:")
single_grain_label.grid(row=11, column=0, padx=2, pady=2)
single_grain_entry = tk.Entry(root)
single_grain_entry.insert(0, preset_values.get("single_grain", ""))
single_grain_entry.grid(row=11, column=1, padx=2, pady=2)

# geometry:
differential_label = tk.Label(
    root, text="Differential Method", font=("Arial", 10, "bold")
)
differential_label.grid(row=12, column=0, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
differential_method_label = tk.Label(root, text="differential_method:")
differential_method_label.grid(row=13, column=0, padx=2, pady=2)
differential_method_entry = tk.Entry(root)
differential_method_entry.insert(0, preset_values.get("differential_method", ""))
differential_method_entry.grid(row=13, column=1, padx=2, pady=2)

# geometry:
integral_label = tk.Label(root, text="Integral Method", font=("Arial", 10, "bold"))
integral_label.grid(row=1, column=2, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
integral_method_label = tk.Label(root, text="integralmethod:")
integral_method_label.grid(row=2, column=2, padx=2, pady=2)
integral_method_entry = tk.Entry(root)
integral_method_entry.insert(0, preset_values.get("integral_method", ""))
integral_method_entry.grid(row=2, column=3, padx=2, pady=2)

# geometry:
temp_label = tk.Label(root, text="Temperature", font=("Arial", 10, "bold"))
temp_label.grid(row=3, column=2, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
Tk_label = tk.Label(root, text="Tk:")
Tk_label.grid(row=4, column=2, padx=2, pady=2)
Tk_entry = tk.Entry(root)
Tk_entry.insert(0, preset_values.get("Tk", ""))
Tk_entry.grid(row=4, column=3, padx=2, pady=2)

# geometry:
initial_conditions_label = tk.Label(
    root, text="Initial Conditions", font=("Arial", 10, "bold")
)
initial_conditions_label.grid(row=5, column=2, columnspan=2, padx=2, pady=2)

# Additional back_encap_thick input with preset value
ini_charge_state_label = tk.Label(root, text="ini_charge_state:")
ini_charge_state_label.grid(row=6, column=2, padx=2, pady=2)
ini_charge_state_entry = tk.Entry(root)
ini_charge_state_entry.insert(0, preset_values.get("ini_charge_state", ""))
ini_charge_state_entry.grid(row=6, column=3, padx=2, pady=2)

ini_potential_label = tk.Label(root, text="ini_potential:")
ini_potential_label.grid(row=7, column=2, padx=2, pady=2)
ini_potential_entry = tk.Entry(root)
ini_potential_entry.insert(0, preset_values.get("ini_potential", ""))
ini_potential_entry.grid(row=7, column=3, padx=2, pady=2)

# geometry:
material_properties_label = tk.Label(
    root, text="Material Properties", font=("Arial", 10, "bold")
)
material_properties_label.grid(row=8, column=2, columnspan=2, padx=2, pady=2)

c_max_label = tk.Label(root, text="c_max:")
c_max_label.grid(row=9, column=2, padx=2, pady=2)
c_max_entry = tk.Entry(root)
c_max_entry.insert(0, preset_values.get("c_max", ""))
c_max_entry.grid(row=9, column=3, padx=2, pady=2)

E_label = tk.Label(root, text="E")
E_label.grid(row=10, column=2, padx=2, pady=2)
E_entry = tk.Entry(root)
E_entry.insert(0, preset_values.get("E", ""))
E_entry.grid(row=10, column=3, padx=2, pady=2)

nu_label = tk.Label(root, text="nu")
nu_label.grid(row=11, column=2, padx=2, pady=2)
nu_entry = tk.Entry(root)
nu_entry.insert(0, preset_values.get("nu", ""))
nu_entry.grid(row=11, column=3, padx=2, pady=2)

k_con_label = tk.Label(root, text="k_con")
k_con_label.grid(row=12, column=2, padx=2, pady=2)
k_con_entry = tk.Entry(root)
k_con_entry.insert(0, preset_values.get("k_con", ""))
k_con_entry.grid(row=12, column=3, padx=2, pady=2)

Dx_div_Dy_label = tk.Label(root, text="Dx_div_Dy")
Dx_div_Dy_label.grid(row=13, column=2, padx=2, pady=2)
Dx_div_Dy_entry = tk.Entry(root)
Dx_div_Dy_entry.insert(0, preset_values.get("Dx_div_Dy", ""))
Dx_div_Dy_entry.grid(row=13, column=3, padx=2, pady=2)

j_applied_label = tk.Label(root, text="j_applied")
j_applied_label.grid(row=2, column=4, padx=2, pady=2)
j_applied_entry = tk.Entry(root)
j_applied_entry.insert(0, preset_values.get("j_applied", ""))
j_applied_entry.grid(row=2, column=5, padx=2, pady=2)

c_label = tk.Label(root, text="Support size")
c_label.grid(row=3, column=4, padx=2, pady=2)
c_entry = tk.Entry(root)
c_entry.insert(0, preset_values.get("c", ""))
c_entry.grid(row=3, column=5, padx=2, pady=2)


# # Load an image
# img = tk.PhotoImage(file="Picture1.png")  # Replace "image.png" with the path to your image

# # Display the image
# img_label = tk.Label(root, image=img)
# img_label.grid(row=2, column=6, rowspan=13, columnspan=2, padx=10, pady=10)


# Button to generate text file and execute script
execute_button = tk.Button(
    root,
    text="Generate Text File and Execute Script",
    command=generate_text_and_execute_script,
)
execute_button.grid(row=4, column=4, rowspan=2, columnspan=2, padx=5, pady=5)

# geometry:
log_label = tk.Label(root, text="Log", font=("Arial", 10, "bold"))
log_label.grid(row=6, column=4, columnspan=2, padx=2, pady=2)


# Output text widget to display subprocess output
output_text_widget = scrolledtext.ScrolledText(root, width=40, height=10)
output_text_widget.grid(
    row=7,
    column=4,
    rowspan=4,
    columnspan=2,
    padx=10,
    pady=10,
)

# # Footnote
# footnote_text = "This is a footnote."
# footnote_label = tk.Label(root, text=footnote_text, font=("Arial", 8), fg="gray")
# footnote_label.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="w")


# Label to display result
# result_label = tk.Label(root, text="output")
# result_label.grid(row=11, column = 4,columnspan=2)

# Start GUI main loop
root.mainloop()
