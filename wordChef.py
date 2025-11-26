# worldChef.py
"""
Unifica utilidades de procesamiento de texto con:
- Normalización
- Búsqueda de patrones con RE
- Resumen simple
- Extracción NER
- Palabras clave
- Análisis de sentimiento

Mejoras:
- Entrada de texto manual o desde archivo
- Mejor manejo de errores
- Inicialización única de dependencias
"""

import re
import sys
import os
from collections import Counter
from datetime import datetime

# ----------------------
# Intentos de import
# ----------------------
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

# ----------------------
# Logging de sesión
# ----------------------
class SessionLogger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
            print("📁 Carpeta 'logs' creada automáticamente.")
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
print(f"📝 Sesión iniciada. Logs guardados en: {logger.filename}\n")

# ----------------------
# Entrada de texto
# ----------------------
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

def obtener_texto() -> str:
    for _ in range(3):
        print("Selecciona cómo ingresar el texto:")
        print("1) Escribir directamente")
        print("2) Leer desde archivo")
        opcion = input("> ").strip()
        if opcion == '1':
            return input("Ingresa texto:\n> ")
        elif opcion == '2':
            ruta = input("Ingresa la ruta del archivo de texto:\n> ").strip()
            texto = leer_archivo(ruta)
            if texto is not None:
                return texto
            print("Intenta de nuevo.")
        else:
            print("Opción no válida, se intentará entrada manual.")
    return input("Ingresa texto:\n> ")

def solicitar_texto() -> str:
    print("\n=== Obtener texto ===")
    return obtener_texto()

# ----------------------
# Inicialización de dependencias
# ----------------------
def cargar_modelo_spacy():
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
    nlp = spacy.blank("es")
    try:
        nlp.add_pipe("sentencizer")
    except Exception:
        pass
    return nlp

def inicializar_nltk():
    if nltk is not None:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except Exception:
            nltk.download('punkt')
            nltk.download('stopwords')

def inicializar_sentimiento():
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
CORRECCIONES_COMUNES = {
    "haiga": "haya", "naiden": "nadie", "nadien": "nadie",
    "aserca": "acerca", "enserio": "en serio", "haber": "a ver", "iva": "iba",
}
SUSTANTIVOS_NO_NEUTROS = {
    "casa": "la casa", "persona": "la persona", "gente": "la gente",
    "niño": "el niño", "niña": "la niña", "camisa": "la camisa"
}

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

# ----------------------
# Patrones con RE
# ----------------------
PATRON_FECHAS = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
PATRON_DINERO = r"\b(?:€?\s?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?\s?(?:€|euros|USD|\$)|\$\d+(?:\.\d+)?\b)"
PATRON_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

def encontrar_fechas(texto): return re.findall(PATRON_FECHAS, texto)
def encontrar_dinero(texto): return re.findall(PATRON_DINERO, texto)
def encontrar_correos(texto): return re.findall(PATRON_EMAIL, texto)

# ----------------------
# Resumen simple
# ----------------------
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

# ----------------------
# Extracción NER
# ----------------------
def extraer_entidades(texto, nlp):
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
# Palabras clave
# ----------------------
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

# ----------------------
# Sentimiento
# ----------------------
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

# ----------------------
# Menú principal
# ----------------------
def menu_principal():
    nlp = cargar_modelo_spacy()
    inicializar_nltk()
    clasificador_sentimiento = inicializar_sentimiento()
    while True:
        print("\n=== worldChef: Menú de utilidades ===")
        print("1) Normalizador de texto")
        print("2) Buscar patrones (fechas, dinero, correos)")
        print("3) Resumen simple")
        print("4) Extracción de entidades")
        print("5) Palabras clave")
        print("6) Análisis de sentimiento")
        print("0) Salir")
        opcion = input("Selecciona una opción: ").strip()
        if opcion == '0':
            print("Saliendo.")
            break
        if opcion not in '123456':
            print("Opción no válida, intenta de nuevo.")
            continue

        texto = solicitar_texto()

        if opcion == '1':
            res = normalizador_texto(texto, nlp)
            if res:
                print("\n--- RESULTADOS ---")
                print("Original:", res['original'])
                print("Lematizado:", res['lematizado'])
                print("Sin repeticiones:", res['sin_repeticiones'])
                print("Corregido:", res['corregido'])
                logger.log("Normalizador", texto, res)

        elif opcion == '2':
            fechas = encontrar_fechas(texto)
            dinero = encontrar_dinero(texto)
            correos = encontrar_correos(texto)
            print('\nFechas:', fechas or 'Ninguna')
            print('Dinero:', dinero or 'Ninguno')
            print('Correos:', correos or 'Ninguno')
            logger.log("Patrones", texto, {"Fechas": fechas, "Dinero": dinero, "Correos": correos})

        elif opcion == '3':
            resumen = resumen_simple(texto, n=3, nlp=nlp)
            print('\n--- RESUMEN ---')
            print(resumen)
            logger.log("Resumen", texto, {"Resumen": resumen})

        elif opcion == '4':
            entidades = extraer_entidades(texto, nlp)
            for k, v in entidades.items():
                print(f"{k}: {v if v else 'Ninguno detectado'}")
            logger.log("NER", texto, entidades)

        elif opcion == '5':
            resultado = extraer_palabras_clave(texto, nlp=nlp)
            if resultado:
                print('Top 5 palabras:', resultado['top_5_palabras'])
                print('Sustantivos relevantes:', resultado['sustantivos'])
                print('Verbos principales:', resultado['verbos'])
                logger.log("Palabras clave", texto, resultado)

        elif opcion == '6':
            sentimiento, score, raw = sentimiento_es(texto, clasificador_sentimiento)
            if sentimiento == "Error":
                print(f"Ocurrió un error: {raw}")
            else:
                print(f"Resultado: {sentimiento} (Confianza: {score:.4f}) Estrellas: {raw}")
                logger.log("Sentimiento", texto, {"Sentimiento": sentimiento, "Confianza": f"{score:.4f}", "Etiqueta": raw})

if __name__ == '__main__':
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.\n")
