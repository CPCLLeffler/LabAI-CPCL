import ttkbootstrap as ttk
def equacao(equation):
    root4 = ttk.Toplevel(height=500, width=500)
    root4.title("LabAI - Equação do Modelo")
    root4.rowconfigure(0, weight=5)
    root4.columnconfigure(0, weight=5)
    equationTextbox = ttk.Text(root4)
    equationTextbox.delete("1.0", "end")
    equationTextbox.insert("1.0", equation)
    equationTextbox.config(state="disabled")
    equationTextbox.grid(row=0, column=0, sticky='nsew')