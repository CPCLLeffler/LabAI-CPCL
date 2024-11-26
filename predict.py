import tkinter
import ttkbootstrap as ttk
import numpy as np
selectedKey = None

def previsao(df, regType):
    def setDictKey():
        if selectedKey:
            dropDict[selectedKey] = insertTextbox.get("1.0", "end").strip()
        root2.after(1, setDictKey)

    def addToDict(key, insertTextbox, dropDict):
        selectedKey = key
        insertTextbox.delete("1.0", "end")
        if key in dropDict:
            insertTextbox.insert("1.0", dropDict[key])
        setDictKey()
    def calculate():
        novo_dado = []
        dados_dropdown = []
        global dropDict, model, first, poly_features
        try:
            for key, value in dropDict.items():
                value = value.strip()
                if value == "":
                    tkinter.messagebox.showerror("Erro!", f"O valor {key} está vazio.")
                try:
                    value = float(value)
                except ValueError:
                    tkinter.messagebox.showerror("Erro!", f"Valor flutuante inválido para {key}: {value}")
                novo_dado.append(value)
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
    dropDict = {}
    root2 = ttk.Toplevel(height=40, width=500)
    root2.title("LabAI - Previsão")
    variable = ttk.StringVar(root2)
    root2.resizable(False, False)

    insertTextbox = ttk.Text(root2, height=1, width=10)
    insertTextbox.grid(row=1, column=2, sticky='nsew')
    resultTextbox = ttk.Text(root2, state="disabled", height=1, width=10)
    resultLabel = ttk.Label(root2, text=df.columns[-1])
    options = [""] + list(df.columns[:-1])
    selectDrop = ttk.OptionMenu(root2, variable, *options, command=addToDict(variable.get(), insertTextbox, dropDict))
    calculateButton = ttk.Button(root2, text="Prever", command=calculate)
    calculateButton.grid(row=1, column=3, sticky='nsew')
    calculateButton.config(state="disabled")
    selectDrop.grid(row=1, column=1, sticky='nsew')
    resultTextbox.grid(row=0, column=4, sticky='nsew')
    resultLabel.grid(row=1, column=4, sticky='nsew')
