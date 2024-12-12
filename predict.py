import tkinter
import ttkbootstrap as ttk
import numpy as np
import tkinter.messagebox
import output
selectedKey = None
lastKey = None
dropDict = {}
def previsao(df, regType, model, poly_features):
    global dropDict
    def save(key, lk):
        global lastKey
        if not lk:
            lk = key
            lastKey = lk
        calculateButton.config(state="normal")
        dropDict[lk] = insertTextbox.get("1.0", "end")
        erase(key)
    def erase(key):
        global lastKey
        insertTextbox.delete("1.0", "end")
        insertTextbox.insert("1.0", dropDict[key])
        lastKey = key
    def calculate(key):
        global dropDict
        novo_dado = []
        save(key, key)
        try:
            for key, value in dropDict.items():
                value = value.strip()
                if value == "":
                    tkinter.messagebox.showerror("Erro!", f"O valor {key} está vazio.")
                try:
                    value = float(value)
                except ValueError:
                    tkinter.messagebox.showerror("Erro!", f"Valor flutuante inválido para {key}: {value}")
                novo_dado.append(float(value))
            novo_dado = np.array(novo_dado).reshape(1, -1)
            if regType == "Regressão Polinomial":
                novo_dado = poly_features.transform(novo_dado)
            prediction = model.predict(novo_dado)
            resultTextbox.config(state="normal")
            resultTextbox.delete("1.0", "end")
            resultTextbox.insert("1.0", str(prediction))
            resultTextbox.config(state="disabled")
            return prediction
        except ValueError as e:
            tkinter.messagebox.showerror("Erro!", f"Erro de Valor (ValueError): {e}")
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")


    root2 = ttk.Toplevel(height=40, width=500)
    root2.title("LabAI - Previsão")
    variable = ttk.StringVar(root2)
    root2.resizable(False, False)
    for key in list(df.columns[:-1]):
        dropDict[key] = ""
    insertTextbox = ttk.Text(root2, height=1, width=10)
    insertTextbox.grid(row=1, column=2, sticky='nsew')
    resultTextbox = ttk.Text(root2, state="disabled", height=1, width=10)
    resultLabel = ttk.Label(root2, text=df.columns[-1])
    options = [""] + list(df.columns[:-1])
    
    selectDrop = ttk.OptionMenu(root2, variable, *options, command=lambda x: save(x, lastKey))
    calculateButton = ttk.Button(root2, text="Prever", command=lambda: calculate(lastKey))
    calculateButton.grid(row=1, column=3, sticky='nsew')
    calculateButton.config(state="disabled")

    selectDrop.grid(row=1, column=1, sticky='nsew')
    resultTextbox.grid(row=0, column=4, sticky='nsew')
    resultLabel.grid(row=1, column=4, sticky='nsew')

    # Add column and row configure for resizing
    for i in range(5):
        root2.columnconfigure(i, weight=1)
    root2.rowconfigure(1, weight=1)
    

