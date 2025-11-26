import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import re
from collections import Counter
from datetime import datetime

try:
    import spacy
except ImportError:
    spacy = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# ---------------------- Logging ----------------------
class SessionLogger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/session_{timestamp}.log"
        self._write_header()

    def _write_header(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"  SESIÓN wordChef - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

    def log(self, tipo: str, entrada: str, resultado):
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {tipo}\n")
            f.write("-"*80 + "\n")
            entrada_truncada = entrada[:100] + ('...' if len(entrada) > 100 else '')
            f.write(f"Entrada: {entrada_truncada}\n\n")
            if isinstance(resultado, dict):
                for clave, valor in resultado.items():
                    f.write(f"  {clave}:\n")
                    if isinstance(valor, (list, set)):
                        for item in valor:
                            f.write(f"    • {item}\n")
                    else:
                        f.write(f"    {valor}\n")
            else:
                f.write(f"Resultado: {resultado}\n")
            f.write("\n" + "="*80 + "\n\n")

logger = SessionLogger()

# ---------------------- Funciones de NLP ----------------------
CORRECCIONES_COMUNES = {
    "haiga": "haya", "naiden": "nadie", "nadien": "nadie",
    "aserca": "acerca", "enserio": "en serio", "haber": "a ver", "iva": "iba",
}
SUSTANTIVOS_NO_NEUTROS = {
    "casa": "la casa", "persona": "la persona", "gente": "la gente",
    "niño": "el niño", "niña": "la niña", "camisa": "la camisa"
}

def cargar_modelo_spacy():
    if spacy is None:
        return None
    modelos = ["es_core_news_sm", "es_core_news_md", "xx_sent_ud_sm"]
    for m in modelos:
        try:
            nlp = spacy.load(m)
            if "sentencizer" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("sentencizer")
                except:
                    pass
            return nlp
        except:
            continue
    nlp = spacy.blank("es")
    try:
        nlp.add_pipe("sentencizer")
    except:
        pass
    return nlp

def inicializar_nltk():
    if nltk is not None:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except:
            nltk.download('punkt')
            nltk.download('stopwords')

def inicializar_sentimiento():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment")
    except:
        return None

def corregir_palabras(doc):
    if doc is None:
        return ""
    corregido = []
    for i, token in enumerate(doc):
        palabra = token.text.lower()
        if palabra in CORRECCIONES_COMUNES:
            corregido.append(CORRECCIONES_COMUNES[palabra])
            continue
        if palabra in SUSTANTIVOS_NO_NEUTROS:
            corregido.append(SUSTANTIVOS_NO_NEUTROS[palabra])
            continue
        if i > 0 and palabra == doc[i-1].text.lower():
            continue
        corregido.append(token.text)
    return " ".join(corregido)

def normalizador_texto(texto, nlp):
    if not texto or len(texto.strip()) == 0:
        return None
    if nlp is None:
        palabras = texto.split()
        sin_repeticiones = " ".join([palabras[i] for i in range(len(palabras)) if i == 0 or palabras[i].lower() != palabras[i-1].lower()])
        return {"original": texto, "lematizado": "(spaCy requerido)", "sin_repeticiones": sin_repeticiones, "corregido": "(spaCy requerido)"}
    doc = nlp(texto)
    lematizado = " ".join([t.lemma_ for t in doc])
    palabras = texto.split()
    sin_repeticiones = " ".join([palabras[i] for i in range(len(palabras)) if i == 0 or palabras[i].lower() != palabras[i-1].lower()])
    texto_corregido = corregir_palabras(doc)
    return {"original": texto, "lematizado": lematizado, "sin_repeticiones": sin_repeticiones, "corregido": texto_corregido}

PATRON_FECHAS = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
PATRON_DINERO = r"\b(?:€?\s?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?\s?(?:€|euros|USD|\$)|\$\d+(?:\.\d+)?\b)"
PATRON_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

def encontrar_fechas(texto): return re.findall(PATRON_FECHAS, texto)
def encontrar_dinero(texto): return re.findall(PATRON_DINERO, texto)
def encontrar_correos(texto): return re.findall(PATRON_EMAIL, texto)

def resumen_simple(texto, n=3, nlp=None):
    if not texto or not texto.strip():
        return "Error: texto vacío."
    oraciones = list(nlp(texto).sents) if nlp else [s.strip() for s in texto.split('.') if s.strip()]
    if len(oraciones) <= n:
        return texto
    puntuaciones = []
    for i, oracion in enumerate(oraciones):
        puntaje = 0
        tokens = oracion if nlp else re.findall(r"\w+", oracion)
        if nlp:
            sustantivos = [t for t in oracion if t.pos_ == "NOUN"]
            puntaje += len(sustantivos)
            longitud = len(oracion.text)
        else:
            sustantivos = [w for w in tokens if len(w) > 2]
            puntaje += len(sustantivos)
            longitud = len(oracion)
        puntaje -= longitud / 200.0
        if i == 0:
            puntaje += 1
        puntuaciones.append((i, puntaje))
    mejores = sorted(puntuaciones, key=lambda x: x[1], reverse=True)[:n]
    indices = sorted([idx for idx, _ in mejores])
    return " ".join(oraciones[i].text.strip() if nlp else oraciones[i] for i in indices)

def extraer_entidades(texto, nlp):
    if nlp is None:
        return {}
    doc = nlp(texto)
    def extraer(tipo): return [ent.text for ent in doc.ents if ent.label_ == tipo]
    return {
        'Personas': sorted(set(extraer('PER'))),
        'Lugares': sorted(set(extraer('LOC'))),
        'Empresas': sorted(set(extraer('ORG'))),
        'Fechas': sorted(set(extraer('DATE'))),
        'Cantidades': sorted(set([ent.text for ent in doc.ents if ent.label_ == 'QUANTITY']))
    }

def extraer_palabras_clave(texto, nlp=None):
    if not texto or not texto.strip():
        return None
    tokens_filtrados, sustantivos_relevantes, verbos_principales = [], [], []
    if nltk:
        tokens = word_tokenize(texto.lower())
        stopwords_es = set(stopwords.words('spanish'))
        tokens_filtrados = [t for t in tokens if t.isalnum() and t not in stopwords_es and len(t) > 2]
        top_5 = Counter(tokens_filtrados).most_common(5)
    else:
        top_5 = []
    if nlp:
        doc = nlp(texto)
        sustantivos_relevantes = Counter([t.text for t in doc if t.pos_ == 'NOUN']).most_common(5)
        verbos_principales = Counter([t.text for t in doc if t.pos_ == 'VERB']).most_common(5)
    return {'top_5_palabras': top_5, 'sustantivos': sustantivos_relevantes, 'verbos': verbos_principales}

def sentimiento_es(texto, clasificador):
    if not texto or not texto.strip():
        return "Error: texto vacío.", 0.0, ""
    if clasificador is None:
        return "Error: transformers no instalado.", 0.0, "transformers missing"
    try:
        resultado = clasificador(texto)[0]
        etiqueta = resultado.get('label', '')
        puntuacion = resultado.get('score', 0.0)
        if "1" in etiqueta or "2" in etiqueta:
            sentimiento = "Negativo"
        elif "3" in etiqueta:
            sentimiento = "Neutral"
        else:
            sentimiento = "Positivo"
        return sentimiento, puntuacion, etiqueta
    except Exception as e:
        return "Error", 0.0, str(e)

# ---------------------- Interfaz ----------------------
class WordChefGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WordChef GUI")
        self.root.geometry("900x600")

        # Textos
        self.texto_entrada = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=10)
        self.texto_entrada.pack(padx=10, pady=10, fill=tk.BOTH)

        self.boton_archivo = tk.Button(self.root, text="Cargar desde archivo", command=self.cargar_archivo)
        self.boton_archivo.pack(pady=5)

        self.frame_botones = tk.Frame(self.root)
        self.frame_botones.pack(pady=5)

        botones = [
            ("Normalizar", self.normalizar),
            ("Patrones", self.patrones),
            ("Resumen", self.resumen),
            ("NER", self.ner),
            ("Palabras Clave", self.palabras_clave),
            ("Sentimiento", self.sentimiento)
        ]
        for txt, cmd in botones:
            tk.Button(self.frame_botones, text=txt, command=cmd).pack(side=tk.LEFT, padx=5)

        self.texto_salida = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=20)
        self.texto_salida.pack(padx=10, pady=10, fill=tk.BOTH)

        # Inicializar dependencias
        self.nlp = cargar_modelo_spacy()
        inicializar_nltk()
        self.clasificador_sentimiento = inicializar_sentimiento()

        self.root.mainloop()

    def cargar_archivo(self):
        ruta = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
        if ruta:
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    contenido = f.read()
                self.texto_entrada.delete(1.0, tk.END)
                self.texto_entrada.insert(tk.END, contenido)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo leer el archivo: {e}")

    def obtener_texto(self):
        texto = self.texto_entrada.get(1.0, tk.END).strip()
        if not texto:
            messagebox.showwarning("Aviso", "Ingresa algún texto.")
            return None
        return texto

    def mostrar_resultado(self, titulo, resultado):
        self.texto_salida.delete(1.0, tk.END)
        self.texto_salida.insert(tk.END, f"=== {titulo} ===\n\n")
        if isinstance(resultado, dict):
            for k, v in resultado.items():
                self.texto_salida.insert(tk.END, f"{k}: {v}\n")
        else:
            self.texto_salida.insert(tk.END, str(resultado))

    # ---------------- Funciones ----------------
    def normalizar(self):
        texto = self.obtener_texto()
        if texto:
            res = normalizador_texto(texto, self.nlp)
            self.mostrar_resultado("Normalización", res)
            logger.log("Normalizador", texto, res)

    def patrones(self):
        texto = self.obtener_texto()
        if texto:
            res = {
                "Fechas": encontrar_fechas(texto),
                "Dinero": encontrar_dinero(texto),
                "Correos": encontrar_correos(texto)
            }
            self.mostrar_resultado("Patrones encontrados", res)
            logger.log("Patrones", texto, res)

    def resumen(self):
        texto = self.obtener_texto()
        if texto:
            res = resumen_simple(texto, n=3, nlp=self.nlp)
            self.mostrar_resultado("Resumen", res)
            logger.log("Resumen", texto, {"Resumen": res})

    def ner(self):
        texto = self.obtener_texto()
        if texto:
            res = extraer_entidades(texto, self.nlp)
            self.mostrar_resultado("NER", res)
            logger.log("NER", texto, res)

    def palabras_clave(self):
        texto = self.obtener_texto()
        if texto:
            res = extraer_palabras_clave(texto, nlp=self.nlp)
            self.mostrar_resultado("Palabras Clave", res)
            logger.log("Palabras clave", texto, res)

    def sentimiento(self):
        texto = self.obtener_texto()
        if texto:
            res, score, etiqueta = sentimiento_es(texto, self.clasificador_sentimiento)
            resultado = {"Sentimiento": res, "Confianza": score, "Etiqueta": etiqueta}
            self.mostrar_resultado("Sentimiento", resultado)
            logger.log("Sentimiento", texto, resultado)

if __name__ == '__main__':
    WordChefGUI()
# worldChef_gui_mod.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from worldChef import (
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
    logger
)

class WorldChefGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌟 WorldChef - Procesamiento de Texto")
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # Inicializamos modelos
        self.nlp = cargar_modelo_spacy()
        inicializar_nltk()
        self.clasificador_sentimiento = inicializar_sentimiento()
        
        # Texto de entrada
        self.texto_label = tk.Label(root, text="Introduce tu texto:", font=("Helvetica", 12))
        self.texto_label.pack(pady=10)
        
        self.texto_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=10, font=("Helvetica", 11))
        self.texto_input.pack(padx=10, pady=5)
        
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
        
    def get_texto(self):
        return self.texto_input.get("1.0", tk.END).strip()
    
    def _setup_normalizador(self):
        btn = tk.Button(self.tab_normalizador, text="Normalizar Texto", command=self.run_normalizador, bg="#4CAF50", fg="white")
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
        btn = tk.Button(self.tab_patrones, text="Buscar Patrones", command=self.run_patrones, bg="#2196F3", fg="white")
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
        btn = tk.Button(self.tab_resumen, text="Generar Resumen", command=self.run_resumen, bg="#FF9800", fg="white")
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
        btn = tk.Button(self.tab_ner, text="Extraer Entidades", command=self.run_ner, bg="#9C27B0", fg="white")
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
        btn = tk.Button(self.tab_keywords, text="Extraer Palabras Clave", command=self.run_keywords, bg="#FF5722", fg="white")
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
        btn = tk.Button(self.tab_sentimiento, text="Analizar Sentimiento", command=self.run_sentimiento, bg="#607D8B", fg="white")
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
