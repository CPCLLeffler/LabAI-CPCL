

#internal imports
import output
import erro
import curves
import predict
import equationFile
# import paths
# import behaviour
#internal imports
import os
import argparse
import tkinter.messagebox
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tkinter.filedialog
import tkinter
import tkinter.simpledialog
from tkinter import PhotoImage
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import webbrowser
parser = argparse.ArgumentParser(description="My Application")
parser.add_argument('--license', action='store_true', help='Show the license information')
args = parser.parse_args()
# if args.license:  # Garante que o caminho seja absoluto
#     webbrowser.open(f"file://{os.getcwd()}/LICENSE")
#     exit()
# Initialize global variables used in functions
os.chdir(os.path.dirname(__file__))
polyKernelSettings = {"grau": float}
degree = 1
kernelGlobal = "rbf"
regType: str
before = False
canvases = []
randomVar = 0
first = False
start = False
dropDict = None
file_path = None
insertTextbox = None
startReg = False
resultTextbox = None
start = False
startAfter = False
startAfterReg = False
df: pd.DataFrame
currentReg = None
regType = ""
equation = ""
poly_features = None
model = None

def filedialog():
    global before, file_path,outputFileW, v

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
            outputFileW = open(f"output/outputFile-{v}.txt", "a", encoding="utf-8")
            outputFileW.write(f"Data do output: {dt.datetime.now()}\n")
            break
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        startLabAI(file_path, file_extension)
def startLabAI(file_path, file_extension):
    global df, label, dependent
    output.output("\nDATASET\n", outputFileW=outputFileW)
    try:
        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            output.output(f"Lendo arquivo Excel localizado em {file_path}", outputFileW=outputFileW)
        else:
            output.output(f"Lendo arquivo CSV localizado em {file_path}", outputFileW=outputFileW)
            df = pd.read_csv(file_path)
    except Exception as e:
        output.output(f"Erro ao ler arquivo: {e}", outputFileW=outputFileW)
        tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
    finally:
        label = ttk.Label(root, text="")
        dependent = ttk.Label(root, text="")
        output.output(f"Variáveis independentes: {list(df.columns[:-1])}", outputFileW=outputFileW)
        output.output(f"Variável dependente: {df.columns[-1]}", outputFileW=outputFileW)
        label.config(text=os.path.basename(file_path))
        dependent.config(text="Variável dependente: " + df.columns[-1])
        placeButtons(df)
        menubar.entryconfig(index=2, state="normal")
        menubar.entryconfig(index=3, state="normal")


def change_reg_type(df: pd.DataFrame, typeReg="Regressão Linear", ntrabalhos=100, intercept=True, grau=1, alfa=1.0, maxIterations=1000,
                    tolerancia=0.00001, solucionador="auto", n_estimadores=100, max_depth=None, min_amostras_divisao=2,
                    min_amostras_folha=1, estado_aleatorio_RF=0, kernel="linear", c=1.0, epsilon=0.1, gama='scale',
                    proporcao_l1=0.5):
    global mse, rmse, mae, r2, randomVar, dropDict, model, before, equation, startAfterReg, regType, kernelGlobal, poly_features, indice, indice2
    regType = typeReg
    startAfterReg = True
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    output.output("\nINFORMAÇÕES DA REGRESSÃO\n", outputFileW=outputFileW)

    if typeReg == "Regressão Linear":
        output.output(f"Iniciando regressão linear\nn_jobs = {ntrabalhos}", outputFileW=outputFileW)
        model = LinearRegression(n_jobs=ntrabalhos)
        model.fit(X_train, y_train)
        coef = model.coef_
        intercept = model.intercept_
        equation = f"y = {coef}x + {intercept}"
        tools.entryconfig(index=4, state="normal")
    elif typeReg == "Regressão Polinomial":
        output.output(f"Iniciando regressão polinomial\ndegree = {grau}", outputFileW=outputFileW)
        poly_features = PolynomialFeatures(degree=grau, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        model = LinearRegression()
        X_train = X_train_poly
        X_test = X_test_poly
        model.fit(X_train, y_train)
        intercept = model.intercept_
        equation = f'y = {model.intercept_:.3f}'
        for i, coef in enumerate(model.coef_[1:], 1):  # Skip the first coefficient (intercept)
            term = poly_features.get_feature_names_out(df.columns[:-1])[i]
            equation += f' + ({coef:.3f} * {term})'
        tools.entryconfig(index=4, state="normal")
    elif typeReg == "Regressão de Suporte Vetorial":
        output.output(f"Iniciando regressão de suporte vetorial\nkernel = {kernel}\nC = {c}\nepsilon = {epsilon}\ngamma = {gama}\ndegree = {grau} ", outputFileW=outputFileW)
        kernelGlobal = kernel
        model = SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gama, degree=grau)
        model.fit(X_train, y_train)
        if kernel == 'linear':
            try:
                coef = model.coef_[0]
                intercept = model.intercept_[0]
                equation = f"y = {coef[0]}x + {intercept}"
                tools.entryconfig(index=4, state="normal")
            except:
                pass
        else:
            tools.entryconfig(index=4, state="disabled")

    elif typeReg == "Regressão Elastic Net":
        output.output(f"Iniciando regressão elastic net\nalpha = {alfa}\nl1_ratio = {proporcao_l1}\nmax_iter = {maxIterations}\ntol = {tolerancia})", outputFileW=outputFileW)
        model = ElasticNet(alpha=alfa, l1_ratio=proporcao_l1, max_iter=maxIterations, tol=tolerancia)        
        model.fit(X_train, y_train)
        equation = f"y = {model.coef_[0]}x + {model.intercept_}"
        tools.entryconfig(index=4, state="normal")
    elif typeReg == "Regressão Ridge":
        output.output(f"Iniciando regressão ridge\nalpha = {alfa}\nmax_iterations= {maxIterations}\nsolver= {solucionador}\ntol = {tolerancia})", outputFileW=outputFileW)
        model = Ridge(alpha=alfa, max_iter=maxIterations, solver=solucionador, tol=tolerancia)
        model.fit(X_train, y_train)
        equation = f"y = {model.coef_[0]}x + {model.intercept_}"
        tools.entryconfig(index=4, state="normal")

    elif typeReg == "Floresta Aleatória":
        output.output(f"Iniciando regressão floresta aleatória\nn_estimators = {n_estimadores}\nmax_depth = {max_depth}\nmin_samples_split = {min_amostras_divisao}\nmin_samples_leaf = {min_amostras_folha}\n random_state = {estado_aleatorio_RF}", outputFileW=outputFileW)
        model = RandomForestRegressor(n_estimators=n_estimadores, max_depth=max_depth,
                                      min_samples_split=min_amostras_divisao, min_samples_leaf=min_amostras_folha,
                                      random_state=estado_aleatorio_RF)
        model.fit(X_train, y_train)
        tools.entryconfig(index=4, state="disabled")



    output.output("\nÍNDICES ESTATÍSTICOS\n", outputFileW=outputFileW)


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    indice = ttk.Label(root, font=("Arial",10))
    indice2 = ttk.Label(root, font=("Arial", 10))
    indice.grid(row=0, column=0, sticky='nsew', rowspan=1)
    indice2.grid(row=0, column=1, sticky='w')
    indice.config(
        text=f"Índices Estatísticos:\nR^2: {r2}\nMSE: {mse}\nRMSE: {rmse}\nMAE: {mae}")
    if typeReg not in ["Floresta Aleatória", "Regressão de Suporte Vetorial"]:
        indice2.config(text=f"Coeficiente e Intercepto\nC: {model.coef_}\nI: {model.intercept_}")
    output.output(f'Mean Squared Error: {mse}', outputFileW=outputFileW)
    output.output(f'Root Mean Squared Error: {rmse}', outputFileW=outputFileW)
    output.output(f'Mean Absolute Error: {mae}', outputFileW=outputFileW)
    output.output(f'R^2 Score: {r2}', outputFileW=outputFileW)
    if typeReg not in ["Floresta Aleatória", "Regressão de Suporte Vetorial"]:
        output.output(f'Intercept: {model.intercept_}', outputFileW=outputFileW)
        output.output(f'Coefficients: {model.coef_}', outputFileW=outputFileW)

    correlacao_dependente = df.iloc[:, :-1].corrwith(df.iloc[:, -1], method='spearman').abs()
    correlacao_dependente = correlacao_dependente.sort_values(ascending=False)
    output.output("Variáveis que mais influenciam:", outputFileW=outputFileW)
    output.output(correlacao_dependente, outputFileW=outputFileW)
    variavel_mais_influente = correlacao_dependente.idxmax()
    correlacao_maxima = correlacao_dependente.max()
    output.output(f"Variável que mais influencia em {df.columns[-1]}:", variavel_mais_influente, outputFileW=outputFileW)
    output.output("Correlação máxima:", correlacao_maxima, outputFileW=outputFileW)
    plot(df)
    before = True
    return r2




def plot(df):
    for i in canvases:
        i.destroy()
    plot_scatter(df)    
    plot_spearman_heatmap(df)


def placeButtons(df):
    github = tkinter.Button(root, image=imagem, command=lambda: webbrowser.open("https://github.com/CPCLLeffler/LabAI-CPCL/"), bd=-1)
    label.grid(row=40, column=0, sticky="nsew")
    github.grid(row=40, column=0, sticky="se")
    dependent.grid(row=39, column=0, columnspan=1, sticky="nsew")
    change_reg_type(df)

def plot_scatter(df):
    global model
    fig, ax = plt.subplots(figsize=(2, 3))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if regType == "Regressão Polinomial":
        X_test = poly_features.transform(X_test)

    y_pred = model.predict(X_test)
    ax.scatter(y_test, y_pred, c='yellow', label='Predito vs. Real')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Preditos')
    ax.set_title('Preditos vs. Reais')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=2, column=0, rowspan=18, columnspan=18, sticky='nsew')
    canvas.draw()
    canvases.append(canvas_widget)


def plot_spearman_heatmap(df):
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
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=21, rowspan=18, columnspan=18, sticky='nsew')
    canvas.draw()
    canvases.append(canvas_widget)
root = ttk.Window(themename="solar")
root.title("LabAI")
imagem = PhotoImage(file="media/github.png")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
root.minsize(height=900, width=1200)
try:
    root.state("zoomed")
except:
    pass
def toggle_fullscreen(event=None):
    if root.attributes("-fullscreen"):
        root.attributes("-fullscreen", False)
    else:
        root.attributes("-fullscreen", True)
root.bind('<Control-Return>', toggle_fullscreen)  
root.bind('<F11>', toggle_fullscreen)  
for i in range(40):
    root.grid_columnconfigure(i, weight=1)
    root.grid_rowconfigure(i//2, weight=1)
def modelConfig():
    global regType
    if regType == "Regressão Linear":
        linear(df)
    elif regType == "Regressão Polinomial":
        poly(df)
    elif regType == "Regressão Ridge":
        ridge(df)
    elif regType == "Floresta Aleatória":
        random_forest(df)
    elif regType == "Regressão de Suporte Vetorial":
        svr(df)
    elif regType == "Regressão Elastic Net":
        elastic_net(df)




# Add a button to switch between scatter plot and heatmap

menubar = ttk.Menu(root)
menubar.config(bg="#002B36", fg="#FFFFFF", activebackground="#013947", activeforeground="#FFFFFF")
arquivo = ttk.Menu(menubar, tearoff=0)
arquivo.add_command(label="Abrir", command=lambda: filedialog())
config = ttk.Menu(menubar, tearoff=0)
config.add_checkbutton(label="Tela Cheia", variable=root.attributes("-fullscreen"), command=lambda: toggle_fullscreen())
# config.add_command(label="Comportamento do LabAI", command=lambda: behaviour.behaviour(root))
# config.add_command(label="Caminhos", command=lambda: paths.paths(root))
ajuda = ttk.Menu(menubar, tearoff=0)
# ajuda.add_command(label="Licensa (GPL v3.0)", command=lambda: webbrowser.open(f"file://{os.getcwd()}/LICENSE"))
ajuda.add_command(label="GitHub", command=lambda: webbrowser.open("https://github.com/CPCLLeffler/LabAI-CPCL/"))
regressionMenu = ttk.Menu(menubar, tearoff=0)
regressionMenu.add_command(label='Regressão Linear', command=lambda: change_reg_type(df, typeReg="Regressão Linear"))
regressionMenu.add_command(label='Regressão Polinomial', command=lambda: change_reg_type(df, typeReg="Regressão Polinomial"))
regressionMenu.add_command(label='Regressão de Suporte Vetorial', command=lambda: change_reg_type(df, typeReg="Regressão de Suporte Vetorial"))
regressionMenu.add_command(label='Floresta Aleatória', command=lambda: change_reg_type(df, typeReg="Floresta Aleatória"))
regressionMenu.add_command(label='Regressão Ridge', command=lambda: change_reg_type(df, typeReg="Regressão Ridge"))
regressionMenu.add_command(label='Regressão Elastic Net', command=lambda: change_reg_type(df ,typeReg="Regressão Elastic Net"))
tools = ttk.Menu(menubar, tearoff=0)
tools.add_command(label="Configurações do Modelo Atual", command=lambda: modelConfig())
tools.add_command(label='Calculador de Erro', command=lambda: erro.erro())
tools.add_command(label='Previsão', command=lambda: predict.previsao(df, regType, model, poly_features))
tools.add_command(label="Tabela", command=lambda: createTable(df))
tools.add_command(label='Equação', command=lambda: equationFile.equacao(equation))
tools.add_command(label="Curvas", command=lambda: curves.curveFinder(df))
menubar.add_cascade(label="Arquivo", menu=arquivo)
menubar.add_cascade(label="Configurações", menu=config)
menubar.add_cascade(label="Ferramentas", menu=tools)
menubar.add_cascade(label="Regressões", menu=regressionMenu)
menubar.add_cascade(label="Ajuda", menu=ajuda)
menubar.entryconfig(index=2, state="disabled")
menubar.entryconfig(index=3, state="disabled")


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
        widget.insert(0, "Input inválido")
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
        widget.insert(0, "Input inválido")
        widget.configure(bootstyle="danger")

def linear(df):
    global rootLinear
    rootLinear = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            n_trabalhos = int(nTrabalhos.get())
            change_reg_type(df, typeReg="Regressão Linear", ntrabalhos=n_trabalhos)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def poly(df):
    global rootPoly
    rootPoly = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            grau_val = int(grau.get())
            change_reg_type(df, typeReg="Regressão Polinomial", grau=grau_val)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def ridge(df):
    global rootRidge
    rootRidge = ttk.Toplevel(root, height=160, width=100)

    def save():
        try:
            tolerancia = 10 ** (-toleranciaDigs.index(toleranciaV.get()))
            change_reg_type(df, typeReg="Regressão Ridge", alfa=float(alfa.get()), tolerancia=tolerancia, solucionador=solucionadorV.get())
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def random_forest(df):
    global rootRandomForest
    rootRandomForest = ttk.Toplevel(root, height=250, width=100)

    def save():
        try:
            change_reg_type(df, typeReg="Floresta Aleatória", n_estimadores=int(n_estimadores.get()), max_depth=int(profundidade_maxima.get()), min_amostras_divisao=int(min_amostras_divisao.get()), min_amostras_folha=int(min_amostras_folha.get()), estado_aleatorio_RF=int(estado_aleatorio.get()))
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def elastic_net(df):
    global rootElasticNet
    rootElasticNet = ttk.Toplevel(root, height=200, width=100)

    def save():
        try:
            tolerancia = 10 ** (-toleranciaDigs.index(toleranciaV.get()))
            change_reg_type(df, typeReg="Regressão Elastic Net", alfa=float(alfa.get()), proporcao_l1=float(l1_ratio.get()), maxIterations=int(max_iter.get()), tolerancia=tolerancia)
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def svr(df):
    global rootSVR, kernelV
    rootSVR = ttk.Toplevel(root, height=200, width=100)
    rootSVR.resizable(False, False)

    def save():
        try:
            if kernelV.get() == "poly":
                change_reg_type(df, typeReg="Regressão de Suporte Vetorial", kernel="poly", c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get()), grau=int(polyKernelSettings["grau"]))
            elif kernelV.get() == "sigmoid":
                change_reg_type(df, typeReg="Regressão de Suporte Vetorial", kernel="sigmoid", c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get))
            else:
                change_reg_type(df, typeReg="Regressão de Suporte Vetorial", kernel=kernelV.get(), c=float(C.get()), epsilon=float(epsilon.get()), gama=eval(gama.get()))
        except Exception as e:
            tkinter.messagebox.showerror("Erro!", f"Erro: {e}")
        finally:
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

def kernel_changed(value):
    if value == 'poly':
        poly_button.grid(column=0, row=4)
    else:
        poly_button.grid_remove()


def poly_kernel_settings():
    def savePoly():
        polyKernelSettings["grau"] = int(grau.get())
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
# Função para criar a tabela


def createTable(df):
    def validate_number_float(new_value, widget_name):
        widget = root.nametowidget(widget_name)
        if new_value.replace('.', '', 1).isdigit() or new_value == "":
            widget.configure(bootstyle="success")
            return True
        else:
            widget.configure(bootstyle="danger")
            return False
    rootTable = ttk.Toplevel(root)
    rootTable.title("LabAI - Tabela")
    rootTable.grid_columnconfigure(0, weight=1)
    rootTable.grid_rowconfigure(0, weight=1)
    def startTable(df):
        for widget in rootTable.winfo_children():
            widget.destroy()
        columns = ["Número da Linha"] + list(df.columns)

        # Create a frame to hold the table and scrollbars
        table_frame = ttk.Frame(rootTable)
        table_frame.grid(row=0, column=0, sticky="nsew")

        # Create Treeview widget
        treeview = ttk.Treeview(
            table_frame,
            columns=[str(x).lower() for x in columns],
            show="headings",
            bootstyle="info"
        )

        # Configure columns and headers
        for column in columns:
            treeview.heading(column.lower(), text=column)
            treeview.column(column.lower(), width=150)

        # Insert rows into the Treeview
    # Insert rows into the Treeview
        for i, row in df.iterrows():
            row_data = [i + 1] + row.tolist()  # Add row number (1-based index)
            treeview.insert("", "end", values=row_data)

        # Add vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=treeview.yview, bootstyle="info"
        )
        h_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=treeview.xview, bootstyle="info"
        )
        treeview.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack Treeview and scrollbars
        treeview.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=2, sticky="ns")
        h_scrollbar.grid(row=1, column=1, sticky="ew")

        # Configure the frame to expand with the window
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        menubarTable = ttk.Menu(rootTable)
        menubarTable.config(bg="#002B36", fg="#FFFFFF", activebackground="#013947", activeforeground="#FFFFFF")
        menubarTable.add_command(label="Salvar e reiniciar LabAI", command=lambda: save())
        menubarTable.add_command(label="Adicionar linha ao final", command=lambda: addLine())
        menubarTable.add_command(label="Remover linha", command=lambda: removeLine())
        menubarTable.add_command(label="Aplicar logarítmo à coluna", command=lambda: log())
        def log():
            def applyLog():
                for key, value in df[dropdown_var.get()].items():
                    df.loc[key, dropdown_var.get()] = (np.log(value)) / (np.log(float(base.get())))
                startTable(df)
                rootLog.destroy()
            rootLog = ttk.Toplevel(rootTable)
            vcmdf = (rootLog.register(validate_number_float), '%P', '%W')
            rootLog.title("LabAI - Aplicar logarítmo à coluna")
            rootLog.resizable(False, False)
            base = ttk.Entry(rootLog, validatecommand=vcmdf)
            baseLabel = ttk.Label(rootLog, text="Base do Logarítmo")
            baseLabel.pack()
            base.pack()
            dropdown_var = ttk.StringVar(value=df.columns[0])
            dropdown = ttk.OptionMenu(rootLog, dropdown_var, *[""] + list(df.columns))
            dropdownLabel = ttk.Label(rootLog, text="Coluna afetada")
            dropdownLabel.pack(pady=10)
            dropdown.pack()
            apply = ttk.Button(rootLog, text="Aplicar logarítmo", command=lambda: applyLog())
            apply.pack()
        rootTable.config(menu=menubarTable)

        def removeLine():
            line = tkinter.simpledialog.askinteger(title="Remover Linha", prompt="Remover qual linha?")
            if line > len(df):
                tkinter.messagebox.showerror("Erro!", "Erro: Número da linha a ser removida ultrapassa número real de linhas do arquivo")
            df.drop(index=df.index[line-1], inplace=True)  # Remove row directly from df
            df.reset_index(drop=True, inplace=True)
            startTable(df)
        def addLine():
            df.loc[len(df)] = [""] * len(df.columns)
            startTable(df)  # Append the new row to the DataFrame

        def save():
            file_pathTable = tkinter.filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Arquivo Excel (.xlsx)", "*.xlsx"), ("Arquivo CSV (.csv)", "*.csv")],title="Selecionar arquivo")
            if os.path.splitext(file_pathTable)[1].lower() == ".xlsx":
                df.to_excel(file_pathTable, index=False)
                startLabAI(file_pathTable, ".xlsx")

            elif os.path.splitext(file_pathTable)[1].lower() == ".csv":
                df.to_csv(file_pathTable, index=False)
                startLabAI(file_pathTable, ".csv")
            else:
                return
            rootTable.destroy()
        def edit_window(row_values, item_id):
            rootEdit = ttk.Toplevel(rootTable)
            rootEdit.title("Editar Linha")

            # Dictionary for tracking edits
            edits = {col: val for col, val in zip(columns, row_values)}

            # Dropdown for selecting columns
            dropdown_var = ttk.StringVar(value=columns[1])
            dropdown_menu = ttk.OptionMenu(
                rootEdit, dropdown_var, *columns
            )
            dropdown_menu.pack(pady=10)

            # Entry box for editing cell value
            entry_var = ttk.StringVar(value=row_values[0])
            entry_box = ttk.Entry(rootEdit, textvariable=entry_var)
            entry_box.pack(pady=10)

            def save_edit():
                # Update the edits dictionary with the current entry value
                current_column = dropdown_var.get()
                edits[current_column] = entry_var.get()

                # Update the DataFrame
                row_index = int(item_id.split("I")[-1], 16)  # Assuming Treeview items map directly to DataFrame rows
                df.loc[row_index, current_column] = entry_var.get()

                # Update the Treeview with the new value
                treeview.item(item_id, values=list(edits.values()))

                rootEdit.destroy()
            def on_dropdown_change(*args):
                # Update entry box value based on dropdown selection
                current_column = dropdown_var.get()
                entry_var.set(edits[current_column])

            dropdown_var.trace_add("write", on_dropdown_change)

            # Save button
            save_button = ttk.Button(
                rootEdit, text="Salvar", command=save_edit
            )
            save_button.pack(pady=10)

        # Function to handle double-click
        def on_double_click(event):
            selected_item = treeview.selection()
            if selected_item:
                item_id = selected_item[0]
                row_values = treeview.item(item_id, "values")
                edit_window(row_values, item_id)

        # Bind double-click event to the Treeview
        treeview.bind("<Double-1>", on_double_click)
    startTable(df)
root.config(menu=menubar)
root.mainloop()
# Grid layout for widgets


