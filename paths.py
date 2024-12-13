# import tkinter
# import tkinter.filedialog
# import ttkbootstrap as ttk
# import os
# import yaml

# def paths(root):
#     rootPath = ttk.Toplevel(root)

#     rootPath.resizable(False, False)
#     rootPath.geometry("400x400") 
#     rootPath.grid_columnconfigure(0, weight=1)
#     rootPath.grid_rowconfigure(0, weight=1)
#     with open("config.yaml", mode="r", encoding="utf-8") as file:
#         content = yaml.safe_load(file)
#     for internal_name, dictValues in content.items():
#         for fn, v in [dictValues.values()]:
#             enum = list(enumerate(dictValues.keys()))
#             rownumber = 0
#             for i in range(len(dictValues.keys())):
#                 ttk.Label(rootPath, text=fn).grid(row=(rownumber*3), column=0)
#                 ttk.Label(rootPath, text=v).grid(row=(rownumber*3)+1, column=0)
#                 ttk.Button(rootPath, text="...", command=lambda: askdir()).grid(row=(rownumber*3)+1, column=1)
#                 rownumber += 1  
#             def askdir():
#                 dir = tkinter.filedialog.askdirectory(title=f"Escolha um diretório para {fn}")
#                 for child in rootPath.winfo_children():
#                     if isinstance(child, ttk.Label):
#                         if child.cget("text") == v:
#                             child.configure(text=dir)
#                             with open("config.yaml", mode="w", encoding="utf-8") as file:
#                                 content[internal_name]["value"] = dir
#                                 yaml.safe_dump(content, file)
           


# def generateConfig(*args):
#     content = {}
#     for name, friendly_name, default in args:
#         content[name] = {"friendly_name": friendly_name, "value": default}
#     return content
# def createConfig():
#     if not os.path.isfile("config.yaml"):
#         content = generateConfig(["output", "Caminho dos registros (output)", os.getcwd()], ["last_loaded", "Último arquivo carregado", None])
#         with open("config.yaml", mode="w", encoding="utf-8") as file:
#             yaml.safe_dump(content, file)

        
# createConfig()