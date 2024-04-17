import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk

def convert():
    # clear output
    output_label.config(text="Output")

    # get input
    input = entry.get()

    # convert
    output = input + " is visualized."

    # set output
    output_label.config(text=output)

# window
window = ttk.Window(themename = 'journal')
window.title("[CViz] CCEP Visualizer")
window.geometry("800x600")

# title
title_label = ttk.Label(master=window, text="CCEP Visualizer", font=("Helvetica", 16))
title_label.pack()

# input field
input_frame = ttk.Frame(master=window)
entry = ttk.Entry(master=input_frame)
button = ttk.Button(master=input_frame, text="Visualize", command = convert)
entry.pack(side = 'left', padx = 10)
button.pack(side = 'left')
input_frame.pack(pady = 10)

# output
output_label = ttk.Label(master=window, text="Output")
output_label.pack(pady = 5)

# run
window.mainloop()
