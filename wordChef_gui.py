import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from wordChef import (
    normalizador_texto,
    encontrar_fechas,
    encontrar_dinero,
    encontrar_correos,
    resumen_simple,
    extraer_entidades,
    extraer_palabras_clave,
    sentimiento_es,
    cargar_modelo_spacy,
    inicializar_nltk,
    inicializar_sentimiento,
    logger,
    leer_archivo
)


class WorldChefGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌟 WorldChef - Procesamiento de Texto")
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # ----------------- Estilo moderno para botones -----------------
        style = ttk.Style()
        style.theme_use('clam')  # tema base moderno
        style.configure('TButton', font=('Helvetica', 11, 'bold'), padding=6)
        style.map('TButton',
                  background=[('active', '#45a049')],
                  foreground=[('active', 'white')])
        
        # Inicializamos modelos
        self.nlp = cargar_modelo_spacy()
        inicializar_nltk()
        self.clasificador_sentimiento = inicializar_sentimiento()
        
        # Frame superior: Entrada de texto y cargar archivo
        frame_top = tk.Frame(root)
        frame_top.pack(pady=10, padx=10, fill="x")
        
        self.texto_label = tk.Label(frame_top, text="Introduce tu texto:", font=("Helvetica", 12))
        self.texto_label.pack(side="top", anchor="w")
        
        self.texto_input = scrolledtext.ScrolledText(frame_top, wrap=tk.WORD, width=100, height=8, font=("Helvetica", 11))
        self.texto_input.pack(pady=5)
        
        # Botones: Cargar + Limpiar Todo
        frame_botones = tk.Frame(frame_top)
        frame_botones.pack(pady=5)
        
        self.btn_cargar = tk.Button(frame_botones, text="📂 Cargar archivo", command=self.cargar_archivo,
                                    bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        self.btn_cargar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_limpiar = tk.Button(frame_botones, text="🧹 Limpiar Todo", command=self.limpiar_todo,
                                     bg="#FF5722", fg="white", activebackground="#E64A19", font=("Helvetica", 10, "bold"))
        self.btn_limpiar.pack(side=tk.LEFT)
        
        # Status label para feedback visual
        self.status_label = tk.Label(frame_top, text="✅ Listo", fg="green", font=("Helvetica", 9))
        self.status_label.pack(side="bottom", pady=2)
        
        # Notebook con pestañas
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(expand=True, fill='both', pady=10, padx=10)
        
        # Cada pestaña
        self.tab_normalizador = ttk.Frame(self.tabs)
        self.tab_patrones = ttk.Frame(self.tabs)
        self.tab_resumen = ttk.Frame(self.tabs)
        self.tab_ner = ttk.Frame(self.tabs)
        self.tab_keywords = ttk.Frame(self.tabs)
        self.tab_sentimiento = ttk.Frame(self.tabs)
        
        self.tabs.add(self.tab_normalizador, text="Normalizador")
        self.tabs.add(self.tab_patrones, text="Patrones")
        self.tabs.add(self.tab_resumen, text="Resumen")
        self.tabs.add(self.tab_ner, text="NER")
        self.tabs.add(self.tab_keywords, text="Palabras Clave")
        self.tabs.add(self.tab_sentimiento, text="Sentimiento")
        
        # Botones y resultados
        self._setup_normalizador()
        self._setup_patrones()
        self._setup_resumen()
        self._setup_ner()
        self._setup_keywords()
        self._setup_sentimiento()
    
    # Limpiar Todo
    def limpiar_todo(self):
        self.texto_input.delete("1.0", tk.END)
        areas = [
            self.normalizador_output,
            self.patrones_output,
            self.resumen_output,
            self.ner_output,
            self.keywords_output,
            self.sentimiento_output
        ]
        for area in areas:
            if hasattr(area, 'delete'):
                area.delete("1.0", tk.END)
        self.status_label.config(text="🧹 ¡Todo limpiado!", fg="#FF5722")
        self.root.after(1500, lambda: self.status_label.config(text="✅ Listo", fg="green"))
    
    def get_texto(self):
        return self.texto_input.get("1.0", tk.END).strip()
    
    def cargar_archivo(self):
        ruta = filedialog.askopenfilename(title="Selecciona un archivo de texto", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if ruta:
            contenido = leer_archivo(ruta)
            if contenido:
                self.texto_input.delete("1.0", tk.END)
                self.texto_input.insert(tk.END, contenido)
                self.status_label.config(text="📂 Archivo cargado", fg="#4CAF50")
    
    # --------------------- Pestañas y sus botones ---------------------
    def _setup_normalizador(self):
        btn = tk.Button(self.tab_normalizador, text="Normalizar Texto", command=self.run_normalizador,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.normalizador_output = scrolledtext.ScrolledText(self.tab_normalizador, wrap=tk.WORD, height=15)
        self.normalizador_output.pack(padx=10, pady=5, fill='both')
    
    def run_normalizador(self):
        texto = self.get_texto()
        if not texto:
            messagebox.showwarning("Aviso", "Introduce un texto primero.")
            return
        res = normalizador_texto(texto, self.nlp)
        self.normalizador_output.delete("1.0", tk.END)
        self.normalizador_output.insert(tk.END, f"Original:\n{res['original']}\n\n")
        self.normalizador_output.insert(tk.END, f"Lematizado:\n{res['lematizado']}\n\n")
        self.normalizador_output.insert(tk.END, f"Sin repeticiones:\n{res['sin_repeticiones']}\n\n")
        self.normalizador_output.insert(tk.END, f"Corregido:\n{res['corregido']}\n")
        logger.log("Normalizador", texto, res)
    
    def _setup_patrones(self):
        btn = tk.Button(self.tab_patrones, text="Buscar Patrones", command=self.run_patrones,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.patrones_output = scrolledtext.ScrolledText(self.tab_patrones, wrap=tk.WORD, height=15)
        self.patrones_output.pack(padx=10, pady=5, fill='both')
    
    def run_patrones(self):
        texto = self.get_texto()
        fechas = encontrar_fechas(texto)
        dinero = encontrar_dinero(texto)
        correos = encontrar_correos(texto)
        self.patrones_output.delete("1.0", tk.END)
        self.patrones_output.insert(tk.END, f"Fechas: {fechas or 'Ninguna'}\n")
        self.patrones_output.insert(tk.END, f"Dinero: {dinero or 'Ninguno'}\n")
        self.patrones_output.insert(tk.END, f"Correos: {correos or 'Ninguno'}\n")
        logger.log("Patrones", texto, {"Fechas": fechas, "Dinero": dinero, "Correos": correos})
    
    def _setup_resumen(self):
        btn = tk.Button(self.tab_resumen, text="Generar Resumen", command=self.run_resumen,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.resumen_output = scrolledtext.ScrolledText(self.tab_resumen, wrap=tk.WORD, height=15)
        self.resumen_output.pack(padx=10, pady=5, fill='both')
    
    def run_resumen(self):
        texto = self.get_texto()
        resumen = resumen_simple(texto, n=3, nlp=self.nlp)
        self.resumen_output.delete("1.0", tk.END)
        self.resumen_output.insert(tk.END, resumen)
        logger.log("Resumen", texto, {"Resumen": resumen})
    
    def _setup_ner(self):
        btn = tk.Button(self.tab_ner, text="Extraer Entidades", command=self.run_ner,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.ner_output = scrolledtext.ScrolledText(self.tab_ner, wrap=tk.WORD, height=15)
        self.ner_output.pack(padx=10, pady=5, fill='both')
    
    def run_ner(self):
        texto = self.get_texto()
        entidades = extraer_entidades(texto, self.nlp)
        self.ner_output.delete("1.0", tk.END)
        for k, v in entidades.items():
            self.ner_output.insert(tk.END, f"{k}: {v if v else 'Ninguno detectado'}\n")
        logger.log("NER", texto, entidades)
    
    def _setup_keywords(self):
        btn = tk.Button(self.tab_keywords, text="Extraer Palabras Clave", command=self.run_keywords,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.keywords_output = scrolledtext.ScrolledText(self.tab_keywords, wrap=tk.WORD, height=15)
        self.keywords_output.pack(padx=10, pady=5, fill='both')
    
    def run_keywords(self):
        texto = self.get_texto()
        resultado = extraer_palabras_clave(texto, nlp=self.nlp)
        self.keywords_output.delete("1.0", tk.END)
        self.keywords_output.insert(tk.END, f"Top 5 palabras: {resultado['top_5_palabras']}\n")
        self.keywords_output.insert(tk.END, f"Sustantivos: {resultado['sustantivos']}\n")
        self.keywords_output.insert(tk.END, f"Verbos: {resultado['verbos']}\n")
        logger.log("Palabras clave", texto, resultado)
    
    def _setup_sentimiento(self):
        btn = tk.Button(self.tab_sentimiento, text="Analizar Sentimiento", command=self.run_sentimiento,
                        bg="#4CAF50", fg="white", activebackground="#45a049", font=("Helvetica", 10, "bold"))
        btn.pack(pady=10)
        self.sentimiento_output = scrolledtext.ScrolledText(self.tab_sentimiento, wrap=tk.WORD, height=15)
        self.sentimiento_output.pack(padx=10, pady=5, fill='both')
    
    def run_sentimiento(self):
        texto = self.get_texto()
        sentimiento, score, raw = sentimiento_es(texto, self.clasificador_sentimiento)
        self.sentimiento_output.delete("1.0", tk.END)
        self.sentimiento_output.insert(tk.END, f"Resultado: {sentimiento}\nConfianza: {score:.4f}\nEstrellas: {raw}\n")
        logger.log("Sentimiento", texto, {"Sentimiento": sentimiento, "Confianza": f"{score:.4f}", "Etiqueta": raw})


if __name__ == "__main__":
    root = tk.Tk()
    app = WorldChefGUI(root)
    root.mainloop()
