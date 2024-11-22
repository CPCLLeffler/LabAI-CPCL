
# def speedConfig():
#     global switchLinear, switchPoly, switchRandomForest, switchSVR, switchRidge, switchElasticNet, linearV, polyV, randomV, svrV, ridgeV, elasticNetV, speedSettings
#     rootConfigSearch = ttk.Toplevel(root, height=900, width=1200)
#     rootConfigSearch.resizable(False, False)

#     def validate_number(new_value, widget_name):
#         widget = root.nametowidget(widget_name)
#         if new_value.isdigit() or new_value == "":
#             widget.configure(bootstyle="success")
#             return True
#         else:
#             widget.configure(bootstyle="danger")
#             return False

#     def on_focus_out(event):
#         widget = event.widget
#         if not widget.get().isdigit():
#             widget.delete(0, ttk.END)
#             widget.insert(0, "Input inválido")
#             widget.configure(bootstyle="danger")
#         else:
#             widget.configure(bootstyle="success")

#     def validate_number_float(new_value, widget_name):
#         widget = root.nametowidget(widget_name)
#         if new_value.replace('.', '', 1).isdigit() or new_value == "":
#             widget.configure(bootstyle="success")
#             return True
#         else:
#             widget.configure(bootstyle="danger")
#             return False

#     def on_focus_out_float(event):
#         widget = event.widget
#         try:
#             float(widget.get())
#             widget.configure(bootstyle="success")
#         except ValueError:
#             widget.delete(0, ttk.END)
#             widget.insert(0, "Input inválido")
#             widget.configure(bootstyle="danger")

#     vcmdf = (root.register(validate_number_float), '%P', '%W')
#     vcmd = (root.register(validate_number), '%P', '%W')

#     def turnOnOff(type, state):
#         def switch(object, state):
#             if state:
#                 object.config(state="normal")
#             else:
#                 object.config(state="disabled")
#         match type:
#             case 0:
#                 switch(n_trabalhos, state)
#             case 1:
#                 switch(grau, state)
#             case 2:
#                 switch(n_arvores, state)
#                 switch(prof_maxima, state)
#                 switch(min_amostra, state)
#                 switch(min_divisao, state)
#                 switch(estado_aleatorio, state)
#             case 3:
#                 switch(c, state)
#                 switch(epsilon, state)
#                 switch(gama, state)
#                 switch(grau_max, state)
#                 switch(coef0Poly, state)
#                 switch(coef0Sigmoid, state)
#             case 4:
#                 switch(alfaRidge, state)
#                 switch(max_iteracoesRidge, state)
#             case 5:
#                 switch(alfaElasticNet, state)
#                 switch(proporcao_l1, state)
#                 switch(max_iteracoesElasticNet, state)



#     # BooleanVars
#     linearV = ttk.BooleanVar(value=True)
#     polyV = ttk.BooleanVar(value=True)
#     randomV = ttk.BooleanVar(value=True)
#     svrV = ttk.BooleanVar(value=True)
#     ridgeV = ttk.BooleanVar(value=True)
#     elasticNetV = ttk.BooleanVar(value=True)

#     # Frame for each model
#     frameLinear = ttk.Frame(rootConfigSearch)
#     frameLinear.place(x=20, y=20, width=360, height=200)

#     framePoly = ttk.Frame(rootConfigSearch)
#     framePoly.place(x=20, y=240, width=360, height=200)

#     frameRandomForest = ttk.Frame(rootConfigSearch)
#     frameRandomForest.place(x=20, y=460, width=360, height=400)

#     frameSVR = ttk.Frame(rootConfigSearch)
#     frameSVR.place(x=400, y=20, width=360, height=700)

#     frameRidge = ttk.Frame(rootConfigSearch)
#     frameRidge.place(x=780, y=460, width=360, height=200)

#     frameElasticNet = ttk.Frame(rootConfigSearch)
#     frameElasticNet.place(x=780, y=20, width=360, height=400)

#     # Checkbuttons
#     switchLinear = ttk.Checkbutton(frameLinear, text="Regressão Linear", command=lambda: turnOnOff(0, linearV.get()), variable=linearV)
#     switchLinear.pack(anchor='w')

#     switchPoly = ttk.Checkbutton(framePoly, text="Regressão Polinomial", command=lambda: turnOnOff(1, polyV.get()), variable=polyV)
#     switchPoly.pack(anchor='w')

#     switchRandomForest = ttk.Checkbutton(frameRandomForest, text="Floresta Aleatória", command=lambda: turnOnOff(2, randomV.get()), variable=randomV)
#     switchRandomForest.pack(anchor='w')

#     switchSVR = ttk.Checkbutton(frameSVR, text="Regressão de Suporte Vetorial", command=lambda: turnOnOff(3, svrV.get()), variable=svrV)
#     switchSVR.pack(anchor='w')

#     switchRidge = ttk.Checkbutton(frameRidge, text="Regressão Ridge", command=lambda: turnOnOff(4, ridgeV.get()), variable=ridgeV)
#     switchRidge.pack(anchor='w')

#     switchElasticNet = ttk.Checkbutton(frameElasticNet, text="Regressão Elastic Net", command=lambda: turnOnOff(5, elasticNetV.get()), variable=elasticNetV)
#     switchElasticNet.pack(anchor='w')

#     # Linear Configurations
#     n_trabalhos = ttk.Entry(frameLinear, validate="key", validatecommand=vcmd)
#     n_trabalhosLabel = ttk.Label(frameLinear, text="Núm. de trabalhos máximo")
#     n_trabalhosLabel.pack()
#     n_trabalhos.pack()


#     # Polynomial Configurations
#     grau = ttk.Entry(framePoly, validate="key", validatecommand=vcmd)
#     grauLabel = ttk.Label(framePoly, text="Grau máximo do polinômio")
#     grauLabel.pack()
#     grau.pack()

#     # Random Forest Configurations
#     n_arvores = ttk.Entry(frameRandomForest, validate="key", validatecommand=vcmd)
#     n_arvoresLabel = ttk.Label(frameRandomForest, text="Núm. máximo de Árvores")
#     n_arvoresLabel.pack()
#     n_arvores.pack()

#     prof_maxima = ttk.Entry(frameRandomForest, validate="key", validatecommand=vcmd)
#     prof_maximaLabel = ttk.Label(frameRandomForest, text="Prof. máxima")
#     prof_maximaLabel.pack()
#     prof_maxima.pack()

#     min_divisao = ttk.Entry(frameRandomForest, validate="key", validatecommand=vcmd)
#     min_divisaoLabel = ttk.Label(frameRandomForest, text="Máx. Amostras para divisão")
#     min_divisaoLabel.pack()
#     min_divisao.pack()

#     min_amostra = ttk.Entry(frameRandomForest, validate="key", validatecommand=vcmd)
#     min_amostraLabel = ttk.Label(frameRandomForest, text="Máx. Amostras para folha")
#     min_amostraLabel.pack()
#     min_amostra.pack()

#     estado_aleatorio = ttk.Entry(frameRandomForest, validate="key", validatecommand=vcmd)
#     estado_aleatorioLabel = ttk.Label(frameRandomForest, text="Estado Aleatório Máximo")
#     estado_aleatorioLabel.pack()
#     estado_aleatorio.pack()

#     # SVR Configurations

#     c = ttk.Entry(frameSVR, validate="key", validatecommand=vcmdf)
#     cLabel = ttk.Label(frameSVR, text="C Máximo")
#     cLabel.pack()
#     c.pack()

#     epsilon = ttk.Entry(frameSVR, validate="key", validatecommand=vcmdf)
#     epsilonLabel = ttk.Label(frameSVR, text="Epsilon Máximo")
#     epsilonLabel.pack()
#     epsilon.pack()

#     gama = ttk.Entry(frameSVR, validate="key", validatecommand=vcmdf)
#     gamaLabel = ttk.Label(frameSVR, text="Gama Máximo")
#     gamaLabel.pack()
#     gama.pack()

#     # Polynomial Kernel Settings
#     polyKernelSettings = ttk.Label(frameSVR, text="Configurações Kernel Polinomial")
#     polyKernelSettings.pack()

#     grau_max = ttk.Entry(frameSVR, validate="key", validatecommand=vcmd)
#     grau_maxLabel = ttk.Label(frameSVR, text="Grau Máximo")
#     grau_maxLabel.pack()
#     grau_max.pack()

#     coef0Poly = ttk.Entry(frameSVR, validate="key", validatecommand=vcmdf)
#     coef0PolyLabel = ttk.Label(frameSVR, text="Coeficiente 0 Máximo (Poly)")
#     coef0PolyLabel.pack()
#     coef0Poly.pack()

#     # Sigmoid Kernel Settings
#     sigmoidKernelSettings = ttk.Label(frameSVR, text="Configurações Kernel Sigmoid")
#     sigmoidKernelSettings.pack()

#     coef0Sigmoid = ttk.Entry(frameSVR, validate="key", validatecommand=vcmdf)
#     coef0SigmoidLabel = ttk.Label(frameSVR, text="Coeficiente 0 Máximo(Sigmoid)")
#     coef0SigmoidLabel.pack()
#     coef0Sigmoid.pack()

#     # Ridge Configurations
#     alfaRidge = ttk.Entry(frameRidge, validate="key", validatecommand=vcmdf)
#     alfaRidgeLabel = ttk.Label(frameRidge, text="Alpha")
#     alfaRidgeLabel.pack()
#     alfaRidge.pack()

#     max_iteracoesRidge = ttk.Entry(frameRidge, validate="key", validatecommand=vcmd)
#     max_iteracoesRidgeLabel = ttk.Label(frameRidge, text="Máx. Iterações (Ridge)")
#     max_iteracoesRidgeLabel.pack()
#     max_iteracoesRidge.pack()

#     # Elastic Net Configurations
#     alfaElasticNet = ttk.Entry(frameElasticNet, validate="key", validatecommand=vcmdf)
#     alfaElasticNetLabel = ttk.Label(frameElasticNet, text="Alpha Máx. (Elastic Net)")
#     alfaElasticNetLabel.pack()
#     alfaElasticNet.pack()

#     proporcao_l1 = ttk.Entry(frameElasticNet, validate="key", validatecommand=vcmdf)
#     proporcao_l1Label = ttk.Label(frameElasticNet, text="Proporção L1 Máx.")
#     proporcao_l1Label.pack()
#     proporcao_l1.pack()

#     max_iteracoesElasticNet = ttk.Entry(frameElasticNet, validate="key", validatecommand=vcmd)
#     max_iteracoesElasticNetLabel = ttk.Label(frameElasticNet, text="Máx. Iterações (Elastic Net)")
#     max_iteracoesElasticNetLabel.pack()
#     max_iteracoesElasticNet.pack()


#     configTuple = [{"n_trabalhos": None, "active": True}, 
#                     {"grau": None, "active": True}, 
#                     {"n_arvores": None, "prof_maxima": None, "min_amostra": None, "min_divisao": None, "estado_aleatorio": None, "active": True}, 
#                     {"c": None, "epsilon": None, "gama": None, "grau_max": None, "coef0Poly": None, "coef0Sigmoid": None, "active": True}, 
#                     {"alfaRidge": None, "max_iteracoesRidge": None, "active": True}, 
#                     {"alfaElasticNet": None, "proporcao_l1": None, "max_iteracoesElasticNet": None, "active": True}]
#     entries = [n_trabalhos, grau, n_arvores, prof_maxima, min_divisao, min_amostra, estado_aleatorio, c, epsilon, gama, coef0Poly, coef0Sigmoid, alfaRidge, max_iteracoesRidge, alfaElasticNet, proporcao_l1, max_iteracoesElasticNet]
#     for i in range(6):
#         if not speedSettings[i]["active"]:
#             turnOnOff(i, False)    
#     for n in range(5):
#         for l in speedSettings[n].keys():
#             for i in entries:
#                 if f'{i=}'.split('=')[0] == l:
#                     i.insert(0, str(speedSettings[n][l]))
#     def save():
#         global speedSettings
#         configTuple[0]["n_trabalhos"] = n_trabalhos.get()
#         configTuple[1]["grau"] = grau.get()
#         configTuple[2]["n_arvores"] = n_arvores.get()
#         configTuple[2]["prof_maxima"] = prof_maxima.get()
#         configTuple[2]["min_amostra"] = min_amostra.get()
#         configTuple[2]["min_divisao"] = min_divisao.get()
#         configTuple[2]["estado_aleatorio"] = estado_aleatorio.get()
#         configTuple[3]["c"] = c.get()
#         configTuple[3]["epsilon"] = epsilon.get()
#         configTuple[3]["gama"] = gama.get()
#         configTuple[3]["grau_max"] = grau_max.get()
#         configTuple[3]["coef0Poly"] = coef0Poly.get()
#         configTuple[3]["coef0Sigmoid"] = coef0Sigmoid.get()
#         configTuple[4]["alfaRidge"] = alfaRidge.get()
#         configTuple[4]["max_iteracoesRidge"] = max_iteracoesRidge.get()
#         configTuple[5]["alfaElasticNet"] = alfaElasticNet.get()
#         configTuple[5]["proporcao_l1"] = proporcao_l1.get()
#         configTuple[5]["max_iteracoesElasticNet"] = max_iteracoesElasticNet.get()
#         quebrar = False
#         for i in range(5):
#             if True in configTuple[i].values():
#                 for j in configTuple[i].values():
#                     if j == None or j == "":
#                         print(configTuple)
#                         tkinter.messagebox.showerror("Erro!", "Alguma configuração ficou vazia!")
#                         quebrar = True
#                     if quebrar:
#                         break
#                     for m in range(5):
#                         lista = [linearV, polyV, randomV, svrV, ridgeV, elasticNetV]
#                         configTuple[m]["active"] = lista[m].get()
#                     speedSettings = configTuple
#                     print(speedSettings)
#                     rootConfigSearch.destroy()

#             if quebrar:
#                 break

#     saveButton = ttk.Button(rootConfigSearch, text="Salvar Configurações", command=lambda: save())
#     saveButton.place(x=500, y=850)

# speedSettings = [{"n_trabalhos": None, "active": True}, 
#                     {"grau": None, "active": True}, 
#                     {"n_arvores": None, "prof_maxima": None, "min_amostra": None, "min_divisao": None, "estado_aleatorio": None, "active": True}, 
#                     {"c": None, "epsilon": None, "gama": None, "grau_max": None, "coef0Poly": None, "coef0Sigmoid": None, "active": True}, 
#                     {"alfaRidge": None, "max_iteracoesRidge": None, "active": True}, 
#                     {"alfaElasticNet": None, "proporcao_l1": None, "max_iteracoesElasticNet": None, "active": True}]