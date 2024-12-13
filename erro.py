import tkinter
import ttkbootstrap as ttk
def erroCalc():
    global xmedido, xcalculado, resultTextboxErro
    try:
        xmedidoV = xmedido.get("1.0", "end").strip()
        xcalculadoV = xcalculado.get("1.0", "end").strip()
        xmedidoV = eval(xmedidoV)
        xcalculadoV = eval(xcalculadoV)
        erroV = 1 - (((xmedidoV-xcalculadoV)/xmedidoV) * 100) 
        erroV = str(erroV)
        erroV += "%"
        resultTextboxErro.config(state="normal")
        resultTextboxErro.delete("1.0", "end")
        resultTextboxErro.insert("1.0", erroV)
        resultTextboxErro.config(state="disabled")
    except Exception as e:
        tkinter.messagebox.showerror("Erro!", f"Erro: {e}")


def erro():
    global root3, xcalculado, xmedido, resultTextboxErro
    root3 = ttk.Toplevel(height=60, width=400)
    root3.title("LabAI - Calculador de Porcentagem de Erro")
    root3.resizable(False, False)
    resultTextboxErro = ttk.Text(root3, state="disabled", height=1, width=10)
    xmedido = ttk.Text(root3, height=1, width=10)
    xcalculado = ttk.Text(root3, height=1, width=10)
    xmedidoLabel = ttk.Label(root3, text="X Medido")
    xcalculadoLabel = ttk.Label(root3, text="X Calculado")
    calculateButtonErro = ttk.Button(root3, text="Calcular Erro", command=lambda: erroCalc())
    calculateButtonErro.grid(row=0, column=3, sticky='nsew')
    xcalculado.grid(row=0, column=0, sticky='nsew')
    xcalculadoLabel.grid(row=1, column=0, sticky='nsew')
    resultTextboxErro.grid(row=0, column=2, sticky='nsew')
    xmedido.grid(row=0, column=1, sticky='nsew')
    xmedidoLabel.grid(row=1, column=1, sticky='nsew')
    resultLabelErro = ttk.Label(root3, text="% de Erro")
    resultLabelErro.grid(row=1, column=2, sticky='nsew')