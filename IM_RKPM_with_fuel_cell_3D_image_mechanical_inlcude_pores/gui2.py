import tkinter as tk
from tkinter import filedialog
import os
import subprocess
# from tkinter.ttk import Notebook, Frame, Style, WebBrowserTab
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import subprocess
import os
import tempfile


"""

why the entry box is not showing unless you click it???
create read input file function!!!
why the output is shown in text box once all is finished????
why cannot show pages if from later frame navigated to page at front???
"""


class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("MeshDOM")     # title
        self.geometry("800x800")  # size of the gui window

        # Load preset values from the text file
        self.preset_values = self.load_preset_values("sample_org.yaml")

        self.categories = {
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
                {"label": "single_grain", "entry": None},                
            ],
            "Time Step Setup": [
                {"label": "t", "entry": None},
                {"label": "nt", "entry": None},
            ],
            "Mesh Free Method": [
                {"label": "differential_method", "entry": None},
                {"label": "integral_method", "entry": None},
                {"label": "c", "entry": None},
            ],
            "Diffusion": [
                {"label": "T_k", "entry": None},
                {"label": "c_max", "entry": None},
                {"label": "j_applied", "entry": None},
                {"label": "ini_charge_state", "entry": None},
                {"label": "ini_potential", "entry": None},
                {"label": "k_con", "entry": None},
                {"label": "Dx_div_Dy", "entry": None},
            ],
            "Mechanical": [
                {"label": "E", "entry": None},
                {"label": "nu", "entry": None},
            ],
            "Execution": [
                {"label": "Number of Cores", "entry": None}
            ]
        }

        self.frames = {}   # all frames/ pages, each frame is a page
        self.current_frame = None
        self.index = 0

        self.create_frames()
        # self.show_frame("general")

        
        self.show_frame("Introduction")


    def load_preset_values(self, filename):
        preset_values = {}
        current_category = None
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()  # remove space at the head or tail
                if line.endswith(":"):
                    current_category = line[:-1]   # category, before :
                    preset_values[current_category] = {}
                elif current_category: # if current_category is not none
                    key, value = line.split(":", 1)  # return list of string, the string is seperated by :
                    preset_values[current_category][key.strip()] = value.strip()
        return preset_values  # preset_values is a two layer library, {'category name': [{variablename1: value1}, {}, {}....]}
    

    def create_frames(self):
        # self.frames["Introduction"]=self.create_intro_frame()
        for category in self.categories: # category is the most upper level key, category of each frame, geometry, time....
            self.frames[category] = self.create_category_frame(category, self.categories[category])

    def create_category_frame(self, category_name, entries): # entries is al variables under a category, it is a list, the component of this list is library
        frame = tk.Frame(self, relief="groove", borderwidth=2) #create frame
        frame.focus_set()
        title_label = tk.Label(frame, text=category_name, font=("Helvetica", 20, "bold"))
        title_label.place(relx=.6, rely=0.1,anchor= tk.CENTER) # category name in a single line
        if category_name == "Introduction":
            
            # frame.pack(fill="both", expand=True)
            # Paragraph at the top
            paragraph = """Meshfree Degradation of (electrochemical) Materials\n add a figure????"""
            var_label = tk.Label(frame, text=paragraph, font=("Helvetica", 16, "bold"))
            var_label.place(relx=0.6, rely=0.5, anchor=tk.CENTER)


            # # Add a WebBrowserTab with a web page
            # notebook = Notebook(frame)
            # notebook.grid(row=1, columnspan=2, pady=10)
            # web_browser_tab = WebBrowserTab(notebook)
            # notebook.add(web_browser_tab, text="PVade Website")
            # web_browser_tab.open("https://www.example.com")  # Replace with your desired URL


        else:    
       
            preset_values = self.preset_values.get(category_name, {}) # library:{'xmin': value, 'xmax': value2.....}
            for i, entry in enumerate(entries):
                # if entry.get("toggle"):
                #     entry["toggle"] = tk.StringVar(value=preset_values.get(entry["label"], "Cylinder2D"))
                #     options = ["Cylinder2D", "Cylinder3D", "Flag2D", "Panels2D", "Panels3D"]
                #     tk.Label(frame, text=entry["label"] + ":").grid(row=i + 1, column=0, sticky="e")
                #     tk.OptionMenu(frame, entry["toggle"], *options).grid(row=i + 1, column=1, padx=5, pady=5)
                # else:
                entry_value = preset_values.get(entry["label"], "")
                var_label = tk.Label(frame, text=entry["label"] + ":")
                var_label.place(relx=0.3,rely=0.2+i/10,anchor=tk.CENTER) # label name
                if entry["label"] == 'single_grain':
                    self.selected_option_single_grain = tk.StringVar(frame)
                    self.selected_option_single_grain.set("True")
                    # Create the dropdown menu
                    options = ["True", "Faulse"]
                    entry["entry"] = tk.OptionMenu(frame, self.selected_option_single_grain, *options)
                    entry["entry"].place(relx=0.6,rely=0.2+i/10,anchor=tk.CENTER)
                else:
                    if entry["label"] == 'differential_method':
                        self.selected_option_differential_method = tk.StringVar(frame)
                        self.selected_option_differential_method.set("explicite")
                        # Create the dropdown menu
                        options = ["direct", "implicite"]
                        entry["entry"] = tk.OptionMenu(frame, self.selected_option_differential_method, *options)
                        entry["entry"].place(relx=0.6,rely=0.2+i/10,anchor=tk.CENTER)

                    else:
                        if entry["label"] == 'integral_method':
                            self.selected_option_integral_method = tk.StringVar(frame)
                            self.selected_option_integral_method.set("gauss")
                            # Create the dropdown menu
                            options = ["gauss", "SCNI"]
                            entry["entry"] = tk.OptionMenu(frame, self.selected_option_integral_method, *options)
                            entry["entry"].place(relx=0.6,rely=0.2+i/10,anchor=tk.CENTER)

                
                        else:
                            entry["entry"] = tk.Entry(frame, textvariable=tk.StringVar())
                            entry["entry"].insert(0, entry_value)  # value of label
                            entry["entry"].place(relx=0.6,rely=0.2+i/10,anchor=tk.CENTER)

                
        # create left tabs to navigate pages:

        categorybotton_intro = tk.Button(frame, text="Introduction",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Introduction"), background='blue')
        categorybotton_intro.place(relx=0.1, rely=0.2, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_geo = tk.Button(frame, text="Geometry",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Geometry"), background='blue')
        categorybotton_geo.place(relx=0.1, rely=0.3, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_time = tk.Button(frame, text="Time Step Setup",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Time Step Setup"), background='blue')
        categorybotton_time.place(relx=0.1, rely=0.4, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_mshfreedom = tk.Button(frame, text="Mesh Free Method",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Mesh Free Method"), background='blue')
        categorybotton_mshfreedom.place(relx=0.1, rely=0.5, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_diffu = tk.Button(frame, text="Diffusion",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Diffusion"), background='blue')
        categorybotton_diffu.place(relx=0.1, rely=0.6, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_mech = tk.Button(frame, text="Mechanical",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Mechanical"), background='blue')
        categorybotton_mech.place(relx=0.1, rely=0.7, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)
        categorybotton_ex = tk.Button(frame, text="Execution",font=("Helvetica", 12, "bold"), command=lambda: self.show_frame("Execution"), background='blue')
        categorybotton_ex.place(relx=0.1, rely=0.8, anchor=tk.CENTER, relwidth=0.2, relheight=0.1)

                            
        if category_name == "Execution":
        #     previous_button = tk.Button(frame, text="Previous", command=lambda: self.previous_frame(category_name)).place(relx=0.1, rely=0.9)
        #     # previous_button.grid(row=len(entries) + 2, column=1, columnspan=1)
        #     next_button = tk.Button(frame, text="Next", command=lambda: self.next_frame(category_name)).place(relx=0.8, rely=0.9)
        #     # next_button.grid(row=len(entries) + 2, column=1, columnspan=1)
        # else:
        #     previous_button = tk.Button(frame, text="Previous", command=lambda: self.previous_frame(category_name)).place(relx=0.1, rely=0.9)
        #     # previous_button.grid(row=len(entries) + 2, columnspan=2, pady=10)
            run_button = tk.Button(frame, text="Run", command=self.run)
            run_button.place(relx=0.8, rely=0.2, anchor=tk.CENTER)
            # run_button.grid(row=len(entries) + 2, columnspan=2, pady=10)
            self.output_text = scrolledtext.ScrolledText(frame, width=80, height=40)
            self.output_text.place(relx=0.6, rely=0.6, anchor=tk.CENTER)
            # self.output_text.grid(row=len(entries) + 3, columnspan=2, padx=10, pady=10)

               
        return frame

    def show_frame(self, category):
        if self.current_frame:
            self.current_frame.pack_forget()  #hide the current frame
        self.current_frame = self.frames[category]
        self.current_frame.pack(fill="both", expand=True)  # show the next or previous frame

    # def next_frame(self, current_category):
    #     current_index = list(self.categories.keys()).index(current_category)
    #     next_index = current_index + 1
    #     # if next_index >= len(self.categories):
    #     #     next_index = 0
    #     next_category = list(self.categories.keys())[next_index]
    #     self.show_frame(next_category)

    # def navigate_frame(self, category):
    #     print(category)
    #     category_index = list(self.categories.keys()).index(category)
    #     new_category = list(self.categories.keys())[category_index]
    #     print(new_category)
    #     self.show_frame(new_category)
    
    # def previous_frame(self, current_category):
    #     current_index = list(self.categories.keys()).index(current_category)
    #     previous_index = current_index - 1
    #     if current_index == 0:
    #         previous_index = 0
    #     previous_category = list(self.categories.keys())[previous_index]
    #     self.show_frame(previous_category)

    def run(self):
        filename = "input.yaml"
        with open(filename, 'w') as file:
            for category, entries in self.categories.items():
                if category != "Execution" and category != "Introduction" :
                    file.write(f"{category}:\n")
                    for entry in entries:
                        if entry['entry']:
                            if entry['entry'].get() != "" :
                            
                                file.write(f"  {entry['label']}: {entry['entry'].get()}\n") # get from entry of gui

                        else:
                            if entry['label'] == "differential_method":
                                file.write(f"  {entry['label']}: {self.selected_option_differential_method.get()}\n") # get from entry of gui
                            if entry['label'] == "integral_method":
                                file.write(f"  {entry['label']}: {self.selected_option_integral_method.get()}\n") # get from entry of gui
                            if entry['label'] == "single_grain":
                                file.write(f"  {entry['label']}: {self.selected_option_single_grain.get()}\n") # get from entry of gui
                        
                        
                # file.write("\n")
        print("Data saved to", filename)
        
        # num_cores = self.categories["Execution"][0]["entry"].get() or "4"
        command = ["python", "main.py", "--input", filename]
        # command = ["python", "ns_main.py"]#, "--input", filename]
        
        # Execute ns_main.py from the current directory
        current_dir = os.getcwd()
        print(command)
        # subprocess.run(command)
        current_dir = os.getcwd()

        # Clear previous output
        self.output_text.delete('1.0', tk.END)

        # Redirect subprocess output to the Text widget
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in iter(process.stdout.readline, ""):
            self.output_text.insert(tk.END, line)
            self.output_text.see(tk.END)  # Scroll to the end of the text

        process.stdout.close()
        process.wait()

if __name__ == "__main__":
    app = Application()
    app.mainloop()