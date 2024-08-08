import platform
import subprocess

def install_dependencies():
    system = platform.system()
    distro = ""
    try:
        if system == "Darwin":
            # macOS
            print("Instalando itens necessários para executar o programa.\nComando executado: brew install python-tk")
            subprocess.run(["brew", "install", "python-tk"])
        elif system == "Linux":
            distro = subprocess.check_output(["uname", "-a"]).decode().lower()
            if "arch" in distro or "manjaro" in distro:
                # Arch/Manjaro
                print("Instalando itens necessários para executar o programa.\nComando executado: sudo pacman -S --noconfirm tk")
                subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "tk"])
            elif "debian" in distro or "ubuntu" in distro:
                # Debian/Ubuntu
                print("Instalando itens necessários para executar o programa.\nComando executado: sudo apt-get install -y python3-tk")
                subprocess.run(["sudo", "apt-get", "install", "-y", "python3-tk"])
            elif "fedora" in distro:
                # Fedora
                print("Instalando itens necessários para executar o programa.\nComando executado: sudo dnf install -y python3-tkinter")
                subprocess.run(["sudo", "dnf", "install", "-y", "python3-tkinter"])
            else:
                print("Tipo de Linux não suportado. Use um sistema operacional como Arch, Debian, Ubuntu ou Fedora.")
                input("Digite uma tecla qualquer para fechar...")
                exit()
        elif system == "Windows":
            pass
    except Exception as e:
        print(f"Erro ao instalar dependências no sistema {system} ({distro}): {e}")
install_dependencies()
import tkinter.messagebox
import numpy as np
import os
import pandas as pd
import datetime as dt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tkinter.filedialog
import tkinter
from tkinter import PhotoImage
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import webbrowser
# Initialize global variables used in functions
os.chdir(os.path.dirname(__file__))
def abrir_github():
    webbrowser.open("https://github.com/CPCLLeffler/LabAI-CPCL/")
current_plot = "scatter"
polyKernelSettings = {"grau": float, "coef0": float }
sigmoidKernelSettings = {"coef0": float}
degree = 1
kernelGlobal = "rbf"
regType: str
before = False
sizeVar = 0.2
randomVar = 0
first = False
selectedKey = None
start = False
dropDict = None
insertTextbox = None
startReg = False
resultTextbox = None
start = False
selectedKey = None
startAfter = False
startAfterReg = False
df = None
currentReg = None
regType = ""
equation = ""


def filedialog():
    global before, file_path, df, outputFileW, v

    file_path = tkinter.filedialog.askopenfilename(
        filetypes=[("Arquivos Excel e CSV", "*.xlsx *.xls *.csv")],
        title="Selecionar arquivo"
    )
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists("output"):
        os.mkdir("output")
    v = 1
    while True:
        if os.path.isfile(f"output/outputFile-{v}.txt"):
            v += 1
        else:
            outputFileW = open(f"output/outputFile-{v}.txt", "a")
            outputFileW.write(f"Data do output: {dt.datetime.now()}\n")
            break
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    label.config(text=os.path.basename(file_path))
    dependent.config(text="Variável dependente: " + df.columns[-1])
    before = False
    placeButtons()

def newPrint(*args):
    for s in args:
        outputFileW.write("\n" + str(s))
        print(s)
        outputFileW.flush()

def change_reg_type(typeReg="Regressão Linear", ntrabalhos=100, intercept=True, grau=1, alfa=1.0, maxIterations=1000,
                    tolerancia=0.00001, solucionador="auto", n_estimadores=100, max_depth=None, min_amostras_divisao=2,
                    min_amostras_folha=1, estado_aleatorio_RF=0, kernel="linear", c=1.0, epsilon=0.1, gama='scale',
                    proporcao_l1=0.5, coef0=1):
    global mse, rmse, mae, r2, randomVar, sizeVar, dropDict, model, before, equation, startAfterReg, regType, kernelGlobal, poly_features
    regType = typeReg
    startAfterReg = True
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeVar, random_state=0)

    if typeReg == "Regressão Linear":
        model = LinearRegression(n_jobs=ntrabalhos)
    elif typeReg == "Regressão Polinomial":
        poly_features = PolynomialFeatures(degree=grau, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        model = LinearRegression()
        X_train = X_train_poly
        X_test = X_test_poly
    elif typeReg == "Regressão de Suporte Vetorial":
        kernelGlobal = kernel
        model = SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gama, degree=grau)
    elif typeReg == "Regressão Elastic Net":
        model = ElasticNet(alpha=alfa, l1_ratio=proporcao_l1, max_iter=maxIterations, tol=tolerancia)
    elif typeReg == "Regressão Ridge":
        model = Ridge(alpha=alfa, max_iter=maxIterations, solver=solucionador, tol=tolerancia)
    elif typeReg == "Floresta Aleatória":
        model = RandomForestRegressor(n_estimators=n_estimadores, max_depth=max_depth,
                                      min_samples_split=min_amostras_divisao, min_samples_leaf=min_amostras_folha,
                                      random_state=estado_aleatorio_RF)

    model.fit(X_train, y_train)
    
    if typeReg == "Regressão Polinomial":
        coefs = model.coef_
        intercept = model.intercept_
        equation = f'y = {model.intercept_:.3f}'
        for i, coef in enumerate(model.coef_[1:], 1):  # Skip the first coefficient (intercept)
            term = poly_features.get_feature_names_out(df.columns[:-1])[i]
            equation += f' + ({coef:.3f} * {term})'
        equationButton.grid(row=14, column=0, sticky='nsew')
    elif typeReg == "Regressão de Suporte Vetorial" and kernel == 'linear':
        try:
            coef = model.coef_[0]
            intercept = model.intercept_[0]
            equation = f"y = {coef[0]}x + {intercept}"
            equationButton.grid(row=14, column=0, sticky='nsew')
        except:
            pass
    elif typeReg in ["Regressão Elastic Net", "Regressão Ridge"]:
        equation = f"y = {model.coef_[0]}x + {model.intercept_}"
        equationButton.grid(row=14, column=0, sticky='nsew')

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    indice.config(
        text=f"Índices Estatísticos:\nR-squared: {r2}\nMSE: {mse}\nRMSE: {rmse}\nMAE: {mae}")
    if typeReg != "Floresta Aleatória":
        indice2.config(text=f"Coeficiente e Intercepto\nC: {model.coef_}\nI: {model.intercept_}")
    newPrint(f'Mean Squared Error: {mse}')
    newPrint(f'Root Mean Squared Error: {rmse}')
    newPrint(f'Mean Absolute Error: {mae}')
    newPrint(f'R^2 Score: {r2}')
    if typeReg != "Floresta Aleatória":
        newPrint(f'Intercept: {model.intercept_}')
        newPrint(f'Coefficients: {model.coef_}')

    correlacao_dependente = df.iloc[:, :-1].corrwith(df.iloc[:, -1], method='spearman').abs()
    correlacao_dependente = correlacao_dependente.sort_values(ascending=False)
    newPrint("Variáveis que mais influenciam:")
    newPrint(correlacao_dependente)
    variavel_mais_influente = correlacao_dependente.idxmax()
    correlacao_maxima = correlacao_dependente.max()
    newPrint("Variável que mais influencia no alcance:", variavel_mais_influente)
    newPrint("Correlação máxima:", correlacao_maxima)
    if current_plot == "scatter":
        plot_scatter()
    else:
        plot_spearman_heatmap()
    before = True
    dropDict = {col: '' for col in df.columns}
    return r2






def placeButtons():
    indice.grid(row=0, column=2, sticky='n')
    indice2.grid(row=0, column=4, sticky='nw')
#   speedDrop.grid(row=28, column=0, columnspan=1, sticky='nsew')
    previsaoButton.grid(row=4, column=0, columnspan=2, sticky='nsew')
    erroButton.grid(row=6, column=0, columnspan=2, sticky='nsew')
    regressionDrop.grid(row=8, column=0, columnspan=2, sticky='nsew')
    configButton.grid(row=10, column=0, columnspan=2)
    regressionLabel.grid(row=8, column=2, columnspan=2, sticky='nsew')
    switch_plot_button.grid(row=2, column=0, columnspan=2, sticky='nsew')
    fileSelect.grid(row=12, column=0, columnspan=2, sticky="nsew")
    label.grid(row=12, column=2)
    github.grid(row=40, column=0)

    change_reg_type()
def plot_scatter():
    global model
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("#002B36")
    fig.set_facecolor("#002B36")
    ax.spines["bottom"].set_color("#FFFFFF")
    ax.spines["top"].set_color("#FFFFFF")
    ax.spines["left"].set_color("#FFFFFF")
    ax.spines["right"].set_color("#FFFFFF")
    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')
    ax.yaxis.label.set_color('#FFFFFF')
    ax.xaxis.label.set_color('#FFFFFF')
    ax.title.set_color("#FFFFFF")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeVar, random_state=0)

    if regType == "Regressão Polinomial":
        X_test = poly_features.transform(X_test)
    
    y_pred = model.predict(X_test)
    ax.scatter(y_test, y_pred, c='blue', label='Predito vs Real')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Preditos')
    ax.set_title('Preditos vs Reais')
    ax.legend()
    
    display_plot(fig)


def plot_spearman_heatmap():
    correlacao_spearman = df.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("#002B36")
    fig.set_facecolor("#002B36")
    ax.spines["bottom"].set_color("#FFFFFF")
    ax.spines["top"].set_color("#FFFFFF")
    ax.spines["left"].set_color("#FFFFFF")
    ax.spines["right"].set_color("#FFFFFF")
    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')
    ax.yaxis.label.set_color('#FFFFFF')
    ax.xaxis.label.set_color('#FFFFFF')
    ax.title.set_color("#FFFFFF")

    sns.heatmap(correlacao_spearman, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax, square=True,
                annot_kws={"color": "black"})  # Set the color of the annotations to black
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color='white')

    ax.set_title('Heatmap da Correlação de Spearman')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='white')  # Set the x-tick labels to white
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')  # Set the y-tick labels to white

    # Display the figure in a Tkinter canvas
    display_plot(fig)
def display_plot(fig):
    global canvas
    if 'canvas' in globals():
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky='nsew')
    canvas.draw()
    
def switch_plot():
    global current_plot
    if current_plot == "scatter":
        plot_spearman_heatmap()
        current_plot = "heatmap"
    else:
        plot_scatter()
        current_plot = "scatter"

def setDictKey():
    global dropDict, insertTextbox, selectedKey, root2
    if selectedKey:
        dropDict[selectedKey] = insertTextbox.get("1.0", "end").strip()
    root2.after(1, setDictKey)

def addToDict(key):
    global dropDict, insertTextbox, selectedKey
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
            newPrint(f"Processando chave: {key}, valor: {value}")
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
def erroCalc():
    global xmedido, xcalculado, resultTextboxErro
    try:
        xmedidoV = xmedido.get("1.0", "end").strip()
        xcalculadoV = xcalculado.get("1.0", "end").strip()
        xmedidoV = eval(xmedidoV)
        xcalculadoV = eval(xcalculadoV)
        erroV = (abs(((xmedidoV-xcalculadoV)/xmedidoV)))*100
        erroV = str(erroV)
        erroV += "%"
        resultTextboxErro.config(state="normal")
        resultTextboxErro.delete("1.0", "end")
        resultTextboxErro.insert("1.0", erroV)
        resultTextboxErro.config(state="disabled")
    except Exception as e:
        tkinter.messagebox.showerror("Erro!", f"Erro: {e}")

def erro():
    global root3, xcalculado, xmedido, resultTextboxErro, erroButton
    root3 = ttk.Toplevel(root, height=60, width=400)
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
    root3.protocol('WM_DELETE_WINDOW', lambda: (root3.destroy(), erroButton.config(state="normal")))

def previsao():
    global insertTextbox, resultTextbox, root2, previsaoButton, dropDict
    dropDict = {}
    root2 = ttk.Toplevel(root, height=40, width=500)
    root2.title("LabAI - Previsão")
    variable = ttk.StringVar(root2)
    previsaoButton.config(state="disabled")
    root2.resizable(False, False)

    resultTextbox = ttk.Text(root2, state="disabled", height=1, width=10)
    resultLabel = ttk.Label(root2, text=df.columns[-1])
    options = [""] + list(df.columns[:-1])
    selectDrop = ttk.OptionMenu(root2, variable, *options, command=addToDict)
    insertTextbox = ttk.Text(root2, height=1, width=10)
    calculateButton = ttk.Button(root2, text="Prever", command=calculate)
    calculateButton.grid(row=1, column=3, sticky='nsew')
    insertTextbox.grid(row=1, column=2, sticky='nsew')
    selectDrop.grid(row=1, column=1, sticky='nsew')
    resultTextbox.grid(row=0, column=4, sticky='nsew')
    resultLabel.grid(row=1, column=4, sticky='nsew')
    root2.protocol('WM_DELETE_WINDOW', lambda: (root2.destroy(), previsaoButton.config(state="normal")))

def equacao():
    global equation
    root4 = ttk.Toplevel(root, height=500, width=500)
    root4.title("LabAI - Equação do Modelo")
    root4.rowconfigure(0, weight=5)
    root4.columnconfigure(0, weight=5)
    equationTextbox = ttk.Text(root4)
    equationTextbox.delete("1.0", "end")
    equationTextbox.insert("1.0", equation)
    equationTextbox.config(state="disabled")
    equationTextbox.grid(row=0, column=0, sticky='nsew')
    root4.protocol('WM_DELETE_WINDOW', lambda: (root4.destroy(), equationButton.config(state="normal")))

def validate_number(new_value, widget_name):
    widget = root.nametowidget(widget_name)
    if new_value.isdigit() or new_value == "":
        widget.configure(bootstyle="success")
        return True
    else:
        widget.configure(bootstyle="danger")
        return False

def on_focus_out(event):
    widget = event.widget
    if not widget.get().isdigit():
        widget.delete(0, ttk.END)
        widget.insert(0, "Input Inválido")
        widget.configure(bootstyle="danger")
    else:
        widget.configure(bootstyle="success")

def validate_number_float(new_value, widget_name):
    widget = root.nametowidget(widget_name)
    if new_value.replace('.', '', 1).isdigit() or new_value == "":
        widget.configure(bootstyle="success")
        return True
    else:
        widget.configure(bootstyle="danger")
        return False

def on_focus_out_float(event):
    widget = event.widget
    try:
        float(widget.get())
        widget.configure(bootstyle="success")
    except ValueError:
        widget.delete(0, ttk.END)
        widget.insert(0, "Invalid input")
        widget.configure(bootstyle="danger")

def linear():
    global rootLinear, configButton
    rootLinear = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            n_trabalhos = int(nTrabalhos.get())
            change_reg_type(typeReg="Regressão Linear", ntrabalhos=n_trabalhos)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootLinear.destroy()

    rootLinear.resizable(False, False)
    vcmd = (rootLinear.register(validate_number), '%P', '%W')
    rootLinear.title("LabAI - Configurações de Regressão")
    for i in range(40):
        rootLinear.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootLinear.grid_rowconfigure(i, weight=1)
    nTrabalhos = ttk.Entry(rootLinear, validate="key", validatecommand=vcmd)
    nTrabalhosL = ttk.Label(rootLinear, text="Número de Trabalhos")
    alterarModelo = ttk.Button(rootLinear, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=2)
    nTrabalhosL.grid(column=1, row=1)
    nTrabalhos.bind("<FocusOut>", on_focus_out)
    nTrabalhos.grid(column=0, row=1)
    rootLinear.protocol('WM_DELETE_WINDOW', lambda: (rootLinear.destroy(), configButton.config(state="normal")))

def poly():
    global rootPoly
    rootPoly = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            grau_val = int(grau.get())
            change_reg_type(typeReg="Regressão Polinomial", grau=grau_val)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootPoly.destroy()

    rootPoly.resizable(False, False)
    vcmd = (rootPoly.register(validate_number), '%P', '%W')
    rootPoly.title("LabAI - Configurações de Regressão")
    for i in range(40):
        rootPoly.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootPoly.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootPoly, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=2)
    grau = ttk.Entry(rootPoly, validate="key", validatecommand=vcmd)
    grauL = ttk.Label(rootPoly, text="Grau máximo do polinômio")
    grauL.grid(column=1, row=1)
    grau.bind("<FocusOut>", on_focus_out)
    grau.grid(column=0, row=1)
    rootPoly.protocol('WM_DELETE_WINDOW', lambda: (rootPoly.destroy(), configButton.config(state="normal")))

def ridge():
    global rootRidge
    rootRidge = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            tolerancia = 10 ** (-toleranciaDigs.index(toleranciaV.get()))
            change_reg_type(typeReg="Regressão Ridge", alfa=float(alfa.get()), tolerancia=tolerancia, solucionador=solucionadorV.get())
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootRidge.destroy()

    rootRidge.resizable(False, False)
    vcmdf = (rootRidge.register(validate_number_float), '%P', '%W')
    vcmd = (rootRidge.register(validate_number), '%P', '%W')
    rootRidge.title("LabAI - Configurações de Regressão")
    for i in range(40):
        rootRidge.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootRidge.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootRidge, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=5)
    alfa = ttk.Entry(rootRidge, validate="key", validatecommand=vcmdf)
    alfaL = ttk.Label(rootRidge, text="Alpha")
    alfaL.grid(column=1, row=1)
    alfa.bind("<FocusOut>", on_focus_out_float)
    alfa.grid(column=0, row=1)
    max = ttk.Entry(rootRidge, validate="key", validatecommand=vcmd)
    maxL = ttk.Label(rootRidge, text="Máx. de Iterações")
    maxL.grid(column=1, row=2)
    max.bind("<FocusOut>", on_focus_out)
    max.grid(column=0, row=2)
    toleranciaL = ttk.Label(rootRidge, text="Qtd. de dígitos de precisão")
    toleranciaL.grid(column=1, row=3)
    toleranciaV = ttk.StringVar(value="5 dígitos")
    toleranciaDigs = ["", "1 dígito", "2 dígitos", "3 dígitos", "4 dígitos", "5 dígitos", "6 dígitos", "7 dígitos"]
    toleranciaD = ttk.OptionMenu(rootRidge, toleranciaV, *toleranciaDigs)
    toleranciaD.grid(column=0, row=3)
    solucionadorL = ttk.Label(rootRidge, text="Solucionador")
    solucionadorL.grid(column=1, row=4)
    solucionadorV = ttk.StringVar(value="auto")
    solucionadorO = ["", "auto", "svd", "cholesky", "sparse_cg", "lsqr", "sag", "saga"]
    solucionadorD = ttk.OptionMenu(rootRidge, solucionadorV, *solucionadorO)
    solucionadorD.grid(column=0, row=4)
    rootRidge.protocol('WM_DELETE_WINDOW', lambda: (rootRidge.destroy(), configButton.config(state="normal")))

def random_forest():
    global rootRandomForest
    rootRandomForest = ttk.Toplevel(root, height=250, width=100)

    def save():
        try:
            change_reg_type(typeReg="Floresta Aleatória", n_estimadores=int(n_estimadores.get()), max_depth=int(profundidade_maxima.get()), min_amostras_divisao=int(min_amostras_divisao.get()), min_amostras_folha=int(min_amostras_folha.get()), estado_aleatorio_RF=int(estado_aleatorio.get()))
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootRandomForest.destroy()

    rootRandomForest.resizable(False, False)
    vcmdf = (rootRandomForest.register(validate_number_float), '%P', '%W')
    vcmd = (rootRandomForest.register(validate_number), '%P', '%W')
    rootRandomForest.title("LabAI - Configurações de Regressão com Floresta Aleatória")
    for i in range(40):
        rootRandomForest.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootRandomForest.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootRandomForest, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=6)
    n_estimadores = ttk.Entry(rootRandomForest, validate="key", validatecommand=vcmd)
    n_estimadoresL = ttk.Label(rootRandomForest, text="Número de Árvores")
    n_estimadoresL.grid(column=1, row=0)
    n_estimadores.bind("<FocusOut>", on_focus_out)
    n_estimadores.grid(column=0, row=0)
    profundidade_maxima = ttk.Entry(rootRandomForest, validate="key", validatecommand=vcmd)
    profundidade_maximaL = ttk.Label(rootRandomForest, text="Profundidade Máxima")
    profundidade_maximaL.grid(column=1, row=1)
    profundidade_maxima.bind("<FocusOut>", on_focus_out)
    profundidade_maxima.grid(column=0, row=1)
    min_amostras_divisao = ttk.Entry(rootRandomForest, validate="key", validatecommand=vcmd)
    min_amostras_divisaoL = ttk.Label(rootRandomForest, text="Min. Amostras para Divisão")
    min_amostras_divisaoL.grid(column=1, row=2)
    min_amostras_divisao.bind("<FocusOut>", on_focus_out)
    min_amostras_divisao.grid(column=0, row=2)
    min_amostras_folha = ttk.Entry(rootRandomForest, validate="key", validatecommand=vcmd)
    min_amostras_folhaL = ttk.Label(rootRandomForest, text="Min. Amostras na Folha")
    min_amostras_folhaL.grid(column=1, row=3)
    min_amostras_folha.bind("<FocusOut>", on_focus_out)
    min_amostras_folha.grid(column=0, row=3)
    estado_aleatorio = ttk.Entry(rootRandomForest, validate="key", validatecommand=vcmd)
    estado_aleatorioL = ttk.Label(rootRandomForest, text="Estado Aleatório")
    estado_aleatorioL.grid(column=1, row=4)
    estado_aleatorio.bind("<FocusOut>", on_focus_out)
    estado_aleatorio.grid(column=0, row=4)
    rootRandomForest.protocol('WM_DELETE_WINDOW', lambda: (rootRandomForest.destroy(), configButton.config(state="normal")))

def elastic_net():
    global rootElasticNet
    rootElasticNet = ttk.Toplevel(root, height=200, width=100)

    def save():
        try:
            tolerancia = 10 ** (-toleranciaDigs.index(toleranciaV.get()))
            change_reg_type(typeReg="Regressão Elastic Net", alfa=float(alfa.get()), proporcao_l1=float(l1_ratio.get()), maxIterations=int(max_iter.get()), tolerancia=tolerancia)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootElasticNet.destroy()

    rootElasticNet.resizable(False, False)
    vcmdf = (rootElasticNet.register(validate_number_float), '%P', '%W')
    vcmd = (rootElasticNet.register(validate_number), '%P', '%W')
    rootElasticNet.title("LabAI - Configurações de Regressão Elastic Net")
    for i in range(40):
        rootElasticNet.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootElasticNet.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootElasticNet, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=5)
    alfa = ttk.Entry(rootElasticNet, validate="key", validatecommand=vcmdf)
    alfaL = ttk.Label(rootElasticNet, text="Alpha")
    alfaL.grid(column=1, row=0)
    alfa.bind("<FocusOut>", on_focus_out_float)
    alfa.grid(column=0, row=0)
    l1_ratio = ttk.Entry(rootElasticNet, validate="key", validatecommand=vcmdf)
    l1_ratioL = ttk.Label(rootElasticNet, text="Proporção L1")
    l1_ratioL.grid(column=1, row=1)
    l1_ratio.bind("<FocusOut>", on_focus_out_float)
    l1_ratio.grid(column=0, row=1)
    max_iter = ttk.Entry(rootElasticNet, validate="key", validatecommand=vcmd)
    maxL = ttk.Label(rootElasticNet, text="Máx. de Iterações")
    maxL.grid(column=1, row=2)
    max_iter.bind("<FocusOut>", on_focus_out)
    max_iter.grid(column=0, row=2)
    toleranciaL = ttk.Label(rootElasticNet, text="Qtd. de dígitos de precisão")
    toleranciaL.grid(column=1, row=3)
    toleranciaV = ttk.StringVar(value="5 dígitos")
    toleranciaDigs = ["1 dígito", "2 dígitos", "3 dígitos", "4 dígitos", "5 dígitos", "6 dígitos", "7 dígitos"]
    toleranciaD = ttk.OptionMenu(rootElasticNet, toleranciaV, *toleranciaDigs)
    toleranciaD.grid(column=0, row=3)
    rootElasticNet.protocol('WM_DELETE_WINDOW', lambda: (rootElasticNet.destroy(), configButton.config(state="normal")))

def svr():
    global rootSVR, kernelV
    rootSVR = ttk.Toplevel(root, height=200, width=100)
    rootSVR.resizable(False, False)

    def save():
        try:
            if kernelV.get() == "poly":
                change_reg_type(typeReg="Regressão de Suporte Vetorial", kernel="poly", c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get()), grau=int(polyKernelSettings["grau"]), coef0=float(polyKernelSettings["coef0"]))
            elif kernelV.get() == "sigmoid":
                change_reg_type(typeReg="Regressão de Suporte Vetorial", kernel="sigmoid", c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get()), coef0=float(sigmoidKernelSettings["coef0"]))
            else:
                change_reg_type(typeReg="Regressão de Suporte Vetorial", kernel=kernelV.get(), c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get()))
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
            configButton.config(state="normal")
            rootSVR.destroy()

    vcmdf = (rootSVR.register(validate_number_float), '%P', '%W')
    rootSVR.title("LabAI - Configurações de Regressão SVR")
    for i in range(40):
        rootSVR.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootSVR.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootSVR, text="Salvar Configurações", command=save)
    alterarModelo.grid(column=0, row=7)
    kernelL = ttk.Label(rootSVR, text="Kernel")
    kernelL.grid(column=1, row=0)
    kernelV = ttk.StringVar(value="rbf")
    kernelO = ["", "rbf", "linear", "poly", "sigmoid"]
    kernelD = ttk.OptionMenu(rootSVR, kernelV, *kernelO, command=kernel_changed)
    kernelD.grid(column=0, row=0)
    C = ttk.Entry(rootSVR, validate="key", validatecommand=vcmdf)
    CL = ttk.Label(rootSVR, text="C")
    CL.grid(column=1, row=1)
    C.bind("<FocusOut>", on_focus_out_float)
    C.grid(column=0, row=1)
    epsilon = ttk.Entry(rootSVR, validate="key", validatecommand=vcmdf)
    epsilonL = ttk.Label(rootSVR, text="Epsilon")
    epsilonL.grid(column=1, row=2)
    epsilon.bind("<FocusOut>", on_focus_out_float)
    epsilon.grid(column=0, row=2)
    gama = ttk.Entry(rootSVR, validate="key")
    gamaL = ttk.Label(rootSVR, text="Gama")
    gamaL.grid(column=1, row=3)
    gama.grid(column=0, row=3)
    global poly_button, sigmoid_button
    poly_button = ttk.Button(rootSVR, text="Configurações de Kernel Polinomial", command=poly_kernel_settings)
    sigmoid_button = ttk.Button(rootSVR, text="Configurações de Kernel Sigmoid", command=sigmoid_kernel_settings)
    rootSVR.protocol('WM_DELETE_WINDOW', lambda: (rootSVR.destroy(), configButton.config(state="normal")))

def kernel_changed(value):
    if value == 'poly':
        poly_button.grid(column=0, row=4)
    else:
        poly_button.grid_remove()
    if value == 'sigmoid':
        sigmoid_button.grid(column=0, row=5)
    else:
        sigmoid_button.grid_remove()

def poly_kernel_settings():
    def savePoly():
        polyKernelSettings["grau"] = int(grau.get())
        polyKernelSettings["coef0"] = float(coef0.get())
        rootPolyKernel.destroy()

    global rootPolyKernel
    rootPolyKernel = ttk.Toplevel(root, height=160, width=100)
    rootPolyKernel.resizable(False, False)
    vcmd = (rootPolyKernel.register(validate_number), '%P', '%W')
    vcmdf = (rootPolyKernel.register(validate_number_float), '%P', '%W')
    rootPolyKernel.title("LabAI - Configurações de Kernel Polinomial")
    for i in range(40):
        rootPolyKernel.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootPolyKernel.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootPolyKernel, text="Salvar Configurações", command=savePoly)
    alterarModelo.grid(column=0, row=3)
    grau = ttk.Entry(rootPolyKernel, validate="key", validatecommand=vcmd)
    grauL = ttk.Label(rootPolyKernel, text="Grau do polinômio")
    grauL.grid(column=1, row=1)
    grau.bind("<FocusOut>", on_focus_out)
    grau.grid(column=0, row=1)
    coef0 = ttk.Entry(rootPolyKernel, validate="key", validatecommand=vcmdf)
    coef0L = ttk.Label(rootPolyKernel, text="Coeficiente 0")
    coef0L.grid(column=1, row=2)
    coef0.bind("<FocusOut>", on_focus_out_float)
    coef0.grid(column=0, row=2)

def sigmoid_kernel_settings():
    def saveSigmoid():
        sigmoidKernelSettings["coef0"] = float(coef0.get())
        rootSigmoidKernel.destroy()

    global rootSigmoidKernel
    rootSigmoidKernel = ttk.Toplevel(root, height=160, width=100)
    rootSigmoidKernel.resizable(False, False)
    vcmd = (rootSigmoidKernel.register(validate_number), '%P', '%W')
    vcmdf = (rootSigmoidKernel.register(validate_number_float), '%P', '%W')
    rootSigmoidKernel.title("LabAI - Configurações de Kernel Sigmoid")
    for i in range(40):
        rootSigmoidKernel.grid_columnconfigure(i, weight=1)
    for i in range(20):
        rootSigmoidKernel.grid_rowconfigure(i, weight=1)
    alterarModelo = ttk.Button(rootSigmoidKernel, text="Salvar Configurações", command=saveSigmoid)
    alterarModelo.grid(column=0, row=2)
    coef0 = ttk.Entry(rootSigmoidKernel, validate="key", validatecommand=vcmdf)
    coef0L = ttk.Label(rootSigmoidKernel, text="Coeficiente 0")
    coef0L.grid(column=1, row=1)
    coef0.bind("<FocusOut>", on_focus_out_float)
    coef0.grid(column=0, row=1)


root = ttk.Window(themename="solar")
root.title("LabAI")
imagem = PhotoImage(file="media/github.png")
root.geometry("800x600")
root.minsize(height=600, width=900)


for i in range(40):
    root.grid_columnconfigure(i, weight=1)
for i in range(20):
    root.grid_rowconfigure(i, weight=1)
def modelConfig():
    global regType
    if regType == "Regressão Linear":
        linear()
    elif regType == "Regressão Polinomial":
        poly()
    elif regType == "Regressão Ridge":
        ridge()
    elif regType == "Floresta Aleatória":
        random_forest()
    elif regType == "Regressão de Suporte Vetorial":
        svr()
    elif regType == "Regressão Elastic Net":
        elastic_net()
def stop():
    global stopButton, killThread
    killThread = True
    stopButton.grid_forget()


regressionType = ttk.StringVar(value="Regressão Linear")
regressionLabel = ttk.Label(text="Tipo de Previsão/Regressão")
speedSelect = ttk.StringVar()
speedSelect.set("Rápido (random_state = 10)")
speedOptions = ["", "Rápido (random_state = 10)", "Devagar (random_state = 25)", "Lerdo (random_state = 50)", "Super Lerdo (random_state = 100)"]
speedDrop = ttk.OptionMenu(root, speedSelect, *speedOptions)
fileSelect = ttk.Button(text="Selecionar Arquivo", command=lambda: filedialog())
regressionOptions = ["", "Regressão Linear", "Regressão Polinomial", "Regressão Ridge", "Floresta Aleatória", "Regressão de Suporte Vetorial", "Regressão Elastic Net"]
regressionDrop = ttk.OptionMenu(root, regressionType, *regressionOptions, command=lambda x: (change_reg_type(typeReg=x)))
reg_type = "Regressão Linear"

# Add a button to switch between scatter plot and heatmap
switch_plot_button = ttk.Button(text="Alternar tipo de Gráfico", command=lambda: switch_plot())

configButton = ttk.Button(text="Configurações do Modelo", command=lambda: (modelConfig(), configButton.config(state="disabled")))
indice = ttk.Label(font=("Arial", 8))
indice2 = ttk.Label(font=("Arial", 8))
label = ttk.Label(text="")
dependent = ttk.Label(text="")
github = tkinter.Button(image=imagem, command=lambda: abrir_github(), bd=-1)
def start_thread():
    global bestRandomVar
    bestRandomVar = threading.Thread(target=findBestRandom)
    bestRandomVar.start()

# Grid configuration

fileSelect.grid(row=38, column=16, columnspan=2, sticky='nsew')
label.grid(row=40, column=16, columnspan=1, sticky='nsew')
dependent.grid(row=40, column=0, columnspan=1, sticky='nsew')
# bestRandomVar = threading.Thread(target=findBestRandom)
# sort = ttk.Button(root, text="Encontrar melhor R^2", command=start_thread)
previsaoButton = ttk.Button(text="Previsões", command=lambda: (previsaoButton.config(state="disabled"), previsao()))
equationButton = ttk.Button(text="Equação", command=lambda: (equationButton.config(state="disabled"), equacao()))
erroButton = ttk.Button(text="Calculador de Erro", command=lambda: (erroButton.config(state="disabled"), erro()))
filedialog()
# Grid layout for widgets


root.mainloop()
