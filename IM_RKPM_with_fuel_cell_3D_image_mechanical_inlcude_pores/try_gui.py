from tkinter import messagebox, ttk
import tkinter as tk

# def display_selection():
#     # Get the selected value.
#     selection = combo.get()
#     messagebox.showinfo(
#         message=f"The selected value is: {selection}",
#         title="Selection"
#     )

# main_window = tk.Tk()
# main_window.config(width=300, height=200)
# main_window.title("Combobox")
# combo = ttk.Combobox(
#     state="readonly",
#     values=["Python", "C", "C++", "Java"]
# )
# combo.place(x=50, y=50)
# button = ttk.Button(text="Display selection", command=display_selection)
# button.place(x=50, y=100)
# main_window.mainloop()

preset_values = {}
current_category = None
# if current_category:
#     print('none')
# current_category = 'fdd'
# if current_category:
#     print('yes')


with open('sample_org.yaml', 'r') as file:
    for line in file:
        line = line.strip()
        if line.endswith(":"):
            current_category = line[:-1]
            preset_values[current_category] = {}

        elif current_category:
            key, value = line.split(":", 1)
            preset_values[current_category][key.strip()] = value.strip()

categories = {
            "Introduction": [],
            # "Introduction": [
            #     {"label": "Test Case", "entry": None},
            # ],
            "Geometry": [
                {"label": "x_min", "entry": None},
                {"label": "x_max", "entry": None},
                {"label": "y_min", "entry": None},
                {"label": "y_max", "entry": None},    
                {"label": "num_boundaries", "entry": None},             
            ],
            "Time Step Setup": [
                {"label": "Total simulation time: t", "entry": None},
                {"label": "Number of time steps: nt", "entry": None},
            ],
            "solver": [
                {"label": "dt", "entry": None},
                {"label": "t_final", "entry": None},
                {"label": "save_text_interval", "entry": None},
                {"label": "save_xdmf_interval", "entry": None},
                {"label": "solver1_ksp", "entry": None},
                {"label": "solver2_ksp", "entry": None},
                {"label": "solver3_ksp", "entry": None},
                {"label": "solver4_ksp", "entry": None},
                {"label": "solver1_pc", "entry": None},
                {"label": "solver2_pc", "entry": None},
                {"label": "solver3_pc", "entry": None},
                {"label": "solver4_pc", "entry": None}
            ],
            "fluid": [
                {"label": "u_ref", "entry": None},
                {"label": "initialize_with_inflow_bc", "entry": None},
                {"label": "time_varying_inflow_bc", "entry": None},
                {"label": "rho", "entry": None},
                {"label": "wind_direction", "entry": None},
                {"label": "nu", "entry": None},
                {"label": "dpdx", "entry": None},
                {"label": "turbulence_model", "entry": None},
                {"label": "c_s", "entry": None},
                {"label": "c_w", "entry": None},
                {"label": "bc_y_min", "entry": None},
                {"label": "bc_y_max", "entry": None},
                {"label": "bc_z_min", "entry": None},
                {"label": "bc_z_max", "entry": None},
                {"label": "periodic", "entry": None},
                {"label": "warm_up_time", "entry": None}
            ],
            "structure": [
                {"label": "beta_relaxation", "entry": None},
                {"label": "tube_connection", "entry": None},
                {"label": "motor_connection", "entry": None},
                {"label": "bc_list", "entry": None},
                {"label": "dt", "entry": None},
                {"label": "rho", "entry": None},
                {"label": "elasticity_modulus", "entry": None},
                {"label": "poissons_ratio", "entry": None},
                {"label": "body_force_x", "entry": None},
                {"label": "body_force_y", "entry": None},
                {"label": "body_force_z", "entry": None}
            ],
            "Execution": [
                {"label": "Number of Cores", "entry": None}
            ]
        }



import tkinter as tk
def on_option_select(event):
    selected_option.set(event)
root = tk.Tk()
root.title("Dropdown Menu Example")
root.geometry("400x300")
# Create a StringVar to hold the selected option
selected_option = tk.StringVar()
# Create the dropdown menu
options = ["Option 1", "Option 2", "Option 3", "Option 4"]
dropdown = tk.OptionMenu(root, selected_option, *options)
dropdown.pack(pady=10)
# Add a button to display the selected option
show_button = tk.Button(root, text="Show Selection", command=lambda: on_option_select(selected_option.get()))
show_button.pack()
# Label to display the selected option
result_label = tk.Label(root, text="")
result_label.pack()
root.mainloop()