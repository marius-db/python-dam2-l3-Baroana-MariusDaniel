# worldChef.py
"""
Unifica utilidades de procesamiento de texto con:
- Normalización
- Búsqueda de patrones con RE
- Resumen simple
- Extracción NER
- Palabras clave
- Análisis de sentimiento
"""


import re
import sys
import os
from collections import Counter
from datetime import datetime


# ----------------------
# Intentos de import
# ----------------------
# Trata de importar spaCy; si no está, asigna None para manejo posterior
try:
    import spacy
except ImportError:
    spacy = None

# Trata de importar nltk y módulos necesarios; si no está, asigna None
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None

# Trata de importar pipeline desde transformers; si no está, asigna None
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# ----------------------
# Logging de sesión - Marius
# ----------------------
class SessionLogger:
    """
    Logger de sesión para guardar los resultados de los análisis en un archivo.

    Cada vez que se crea una instancia de esta clase se genera un archivo
    nuevo en la carpeta `logs/` con nombre `session_YYYYMMDD_HHMMSS.log`.

    Métodos principales:
    - log(tipo, entrada, resultado): guarda un análisis genérico.
    - registrar_patron(tipo_patron, coincidencias): guarda búsquedas por patrón.

    El logger mantiene el archivo legible para un humano, con encabezados,
    timestamps y presentación en viñetas para listas. Está pensado para
    seguimiento de sesiones interactivas desde el CLI.
    """
    # Constructor crea carpeta logs si no existe y crea fichero log con timestamp
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
            print("📁 Carpeta 'logs' creada automáticamente.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/session_{timestamp}.log"
        self._write_header()

    # Escribe cabecera inicial del log con formato y fecha
    def _write_header(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"  SESIÓN wordChef - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

    # Añade entrada de log con tipo, fragmento de entrada y resultado
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
                        f.write(f"   {valor}\n")
            else:
                f.write(f"Resultado: {resultado}\n")
            f.write("\n" + "="*80 + "\n\n")


logger = SessionLogger()
print(f"📝 Sesión iniciada. Logs guardados en: {logger.filename}\n")


# ----------------------
# Entrada de texto
# ----------------------
# Función para leer texto desde un archivo; devuelve el contenido o None si error
def leer_archivo(ruta: str) -> str | None:
    if not os.path.exists(ruta):
        print(f"Error: El archivo '{ruta}' no existe.")
        return None
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

# ----------------------
# Inicialización de dependencias
# ----------------------

def cargar_modelo_spacy():
    """
    Carga un modelo de spaCy para procesamiento en español.

    Intenta cargar varios modelos de spaCy en el siguiente orden:
    1. `es_core_news_sm`
    2. `es_core_news_md`
    3. `xx_sent_ud_sm`

    Si ninguno está disponible, crea un pipeline vacío para español (`spacy.blank("es")`)
    e intenta añadir un `sentencizer` para segmentación en oraciones.

    Returns:
        nlp (spacy.lang): Objeto de procesamiento lingüístico de spaCy.
        Si spaCy no está instalado, retorna `None`.
    """
    if spacy is None:
        print("Aviso: spaCy no instalado. Algunas funciones no estarán disponibles.")
        return None

    modelos = ["es_core_news_sm", "es_core_news_md", "xx_sent_ud_sm"]

    for m in modelos:
        try:
            nlp = spacy.load(m)
            if "sentencizer" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("sentencizer")
                except Exception:
                    pass
            return nlp
        except Exception:
            continue

    # Si no cargó ningún modelo, crear pipeline vacío
    nlp = spacy.blank("es")
    try:
        nlp.add_pipe("sentencizer")
    except Exception:
        pass
    return nlp


def inicializar_nltk():
    """
    Inicializa recursos necesarios de NLTK.

    Verifica si los paquetes:
    - `punkt` (tokenizador)
    - `stopwords` (lista de stopwords)
    
    están disponibles. Si no, los descarga automáticamente.

    Este procedimiento solo se ejecuta si NLTK está instalado.

    Returns:
        None
    """
    if nltk is not None:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except Exception:
            nltk.download('punkt')
            nltk.download('stopwords')


def inicializar_sentimiento():
    """
    Inicializa un clasificador de sentimiento usando Transformers.

    Carga el modelo `nlptown/bert-base-multilingual-uncased-sentiment`
    a través del pipeline de HuggingFace.

    Returns:
        pipeline or None:
            - Un pipeline de análisis de sentimiento si Transformers está disponible.
            - `None` si no se puede inicializar o la librería no está instalada.
    """
    if pipeline is None:
        return None

    try:
        return pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    except Exception:
        return None

# ----------------------
# Normalización
# ----------------------
# Diccionario para corregir errores comunes de palabras
CORRECCIONES_COMUNES = {
    "haiga": "haya", "naiden": "nadie", "nadien": "nadie",
    "aserca": "acerca", "enserio": "en serio", "haber": "a ver", "iva": "iba",
}
# Sustantivos que no deben ser neutros, con artículo definido
SUSTANTIVOS_NO_NEUTROS = {
    "casa": "la casa", "persona": "la persona", "gente": "la gente",
    "niño": "el niño", "niña": "la niña", "camisa": "la camisa"
}

# Corrige palabras comunes y evita repeticiones consecutivas en un doc spaCy
def corregir_palabras(doc):
    """
    Corrige errores ortográficos comunes y evita repeticiones consecutivas.
    
    Args:
        doc: Documento spaCy procesado.
    
    Returns:
        str: Texto corregido y sin repeticiones.
    """
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

# Normaliza el texto: lematiza, elimina repeticiones, corrige palabras, usando spaCy si está
def normalizador_texto(texto, nlp):
    """
    Normaliza texto: lematiza, elimina repeticiones y corrige palabras.
    
    Args:
        texto (str): Texto original a procesar.
        nlp: Pipeline spaCy para lematización.
    
    Returns:
        tuple: (original, lematizado, sin_repeticiones, corregido)
    """
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


# ----------------------
# Patrones con RE
# ----------------------
# Expresiones regulares para fechas, dinero y correos electrónicos
PATRON_FECHAS = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
PATRON_DINERO = r"\b(?:€?\s?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?\s?(?:€|euros|USD|\$)|\$\d+(?:\.\d+)?\b)"
PATRON_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

# Funciones para buscar patrones en texto usando re.findall
def encontrar_fechas(texto): return re.findall(PATRON_FECHAS, texto)
"""
    Extrae patrones de fechas del texto usando expresiones regulares.
    
    Args:
        texto (str): Texto a analizar.
    
    Returns:
        list[str]: Lista de fechas encontradas.
    """
def encontrar_dinero(texto): return re.findall(PATRON_DINERO, texto)
"""
    Extrae patrones monetarios (euros, USD) del texto.
    
    Args:
        texto (str): Texto a analizar.
    
    Returns:
        list[str]: Lista de cantidades monetarias.
    """
def encontrar_correos(texto): return re.findall(PATRON_EMAIL, texto)
"""
    Extrae direcciones de email del texto.
    
    Args:
        texto (str): Texto a analizar.
    
    Returns:
        list[str]: Lista de correos electrónicos.
    """

# ----------------------
# Resumen simple
# ----------------------
# Extrae un resumen simple basado en relevancia de oraciones que contienen sustantivos
def resumen_simple(texto, n=3, nlp=None):
    """
    Genera resumen automático extrayendo oraciones más relevantes.
    
    Args:
        texto (str): Texto original.
        n (int): Número máximo de oraciones en resumen.
        nlp: Pipeline spaCy opcional.
    
    Returns:
        str: Resumen conciso del texto.
    """
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


# ----------------------
# Extracción NER
# ----------------------
# Extrae entidades nombradas del texto con spaCy, clasificando en categorías relevantes
def extraer_entidades(texto, nlp):
    """
    Extrae entidades nombradas (NER) clasificadas por tipo.
    
    Args:
        texto (str): Texto a analizar.
        nlp: Pipeline spaCy.
    
    Returns:
        dict: {'Personas': [...], 'Lugares': [...], ...}
    """
    if nlp is None:
        print("Aviso: spaCy no disponible — NER no puede ejecutarse.")
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


# ----------------------
# Palabras clave - Marius
# ----------------------
# Extrae palabras clave usando nltk para filtrar stopwords y spaCy para sustantivos y verbos
def extraer_palabras_clave(texto, nlp=None):
    """
        Extrae palabras clave relevantes de un texto en español.

        Proceso:
        - Usa NLTK para tokenizar y calcular las `top_5_palabras` (eliminando
            stopwords en español y tokens no alfanuméricos).
        - Si se proporciona un objeto `nlp` de spaCy, extrae los sustantivos
            y verbos principales (parte del discurso POS) y devuelve sus
            frecuencias.

        Parámetros:
        - texto (str): texto de entrada. Si está vacío, devuelve `None`.
        - nlp (spaCy Language, opcional): objeto spaCy para análisis morfosintáctico.

        Retorna:
        dict con claves:
            - 'top_5_palabras': lista de tuplas (palabra, frecuencia)
            - 'sustantivos': lista de tuplas (sustantivo, frecuencia)
            - 'verbos': lista de tuplas (verbo, frecuencia)

        Nota:
        - Si NLTK o spaCy no están disponibles, la función hace un "fallback"
            devolviendo listas vacías o usando solo la parte que sí esté disponible.
        """
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
    return {
        'top_5_palabras': top_5, 
        'sustantivos': sustantivos_relevantes, 
        'verbos': verbos_principales
    }


# ----------------------
# Sentimiento
# ----------------------
# Determina sentimiento del texto como positivo, neutral o negativo con modelo transformers
def sentimiento_es(texto, clasificador):
    """
    Analiza el sentimiento usando el modelo multilingual-uncased de Nlptown.

    Retorna:
        - sentimiento: "Positivo", "Negativo", "Neutral" o "Error"
        - score: confianza (0–1)
        - etiqueta_raw: etiqueta original del modelo (ej. "4 stars")
        - valor_continuo: valor normalizado (-1 a 1)
    """
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
