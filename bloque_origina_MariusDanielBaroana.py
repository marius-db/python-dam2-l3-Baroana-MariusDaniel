"""
Extracción de Palabras Clave usando spaCy y NLTK con Logging de Sesión

Este programa extrae palabras clave importantes de un texto en español:
- Top 5 palabras más frecuentes (NLTK)
- Sustantivos relevantes (spaCy)
- Verbos principales (spaCy)

Además, registra los resultados de cada extracción en un archivo de log estructurado.
"""

# ----------------------
# IMPORTACIONES
# ----------------------
import os
from datetime import datetime
from collections import Counter

try:
    import spacy
except ImportError:
    print("Error: spacy no está instalado. Ejecuta: pip install spacy")
    exit()

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Error: nltk no está instalado. Ejecuta: pip install nltk")
    exit()

# ----------------------
# DESCARGA DE RECURSOS NLTK
# ----------------------
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')

# ----------------------
# CARGA DEL MODELO DE SPACY
# ----------------------
try:
    nlp = spacy.load('es_core_news_sm')
except OSError:
    print("Error: Modelo de spaCy no encontrado.")
    print("Por favor ejecuta: python -m spacy download es_core_news_sm")
    exit()

# ----------------------
# CLASS: SessionLogger
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
    timestamps y presentación en viñetas para listas.
    Está pensado para seguimiento de sesiones interactivas desde el CLI.
    """

    def __init__(self):
        """Crea carpeta logs si no existe y genera el archivo de log con timestamp."""
        if not os.path.exists("logs"):
            os.makedirs("logs")
            print("📁 Carpeta 'logs' creada automáticamente.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/session_{timestamp}.log"
        self._write_header()

    def _write_header(self):
        """Escribe cabecera inicial del log con formato y fecha actual."""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  SESIÓN de Extracción de Palabras Clave - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, tipo: str, entrada: str, resultado):
        """
        Añade una entrada al log con tipo, fragmento de entrada y resultado.

        Args:
            tipo (str): Tipo de acción registrada (e.g., “Extracción de palabras”)
            entrada (str): Texto procesado
            resultado: Resultado del análisis (dict o str)
        """
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {tipo}\n")
            f.write("-" * 80 + "\n")
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
            f.write("\n" + "=" * 80 + "\n\n")


logger = SessionLogger()
print(f"📝 Sesión iniciada. Logs guardados en: {logger.filename}\n")

# ----------------------
# FUNCIÓN: extraer_palabras_clave
# ----------------------
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
        texto (str): texto de entrada. Si está vacío, devuelve `None`.
        nlp (spaCy Language, opcional): objeto spaCy para análisis morfosintáctico.

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

    # Extracción de top palabras con NLTK
    if nltk:
        tokens = word_tokenize(texto.lower())
        stopwords_es = set(stopwords.words('spanish'))
        tokens_filtrados = [t for t in tokens if t.isalnum() and t not in stopwords_es and len(t) > 2]
        top_5 = Counter(tokens_filtrados).most_common(5)
    else:
        top_5 = []

    # Extracción de sustantivos y verbos con spaCy
    if nlp:
        doc = nlp(texto)
        sustantivos_relevantes = Counter([t.text for t in doc if t.pos_ == 'NOUN']).most_common(5)
        verbos_principales = Counter([t.text for t in doc if t.pos_ == 'VERB']).most_common(5)

    resultado = {
        'top_5_palabras': top_5,
        'sustantivos': sustantivos_relevantes,
        'verbos': verbos_principales
    }

    # Registrar en log
    logger.log("Extracción de Palabras Clave", texto, resultado)

    return resultado

# ----------------------
# FUNCIÓN: mostrar_resultados
# ----------------------
def mostrar_resultados(palabras_clave):
    """
    Muestra los resultados de forma clara y organizada en consola.

    Args:
        palabras_clave (dict): Diccionario con los resultados de la extracción.
    """
    if not palabras_clave:
        print("Error: No hay resultados para mostrar.")
        return

    print("\n" + "=" * 50)
    print("RESULTADOS DE EXTRACCIÓN DE PALABRAS CLAVE")
    print("=" * 50)

    print("\nTOP 5 PALABRAS MÁS FRECUENTES:")
    for i, (palabra, freq) in enumerate(palabras_clave['top_5_palabras'], 1):
        print(f"  {i}. {palabra}: {freq} veces")

    print("\nSUSTANTIVOS RELEVANTES:")
    for i, (sustantivo, freq) in enumerate(palabras_clave['sustantivos'], 1):
        print(f"  {i}. {sustantivo}: {freq} veces")

    print("\nVERBOS PRINCIPALES:")
    for i, (verbo, freq) in enumerate(palabras_clave['verbos'], 1):
        print(f"  {i}. {verbo}: {freq} veces")

    print("\n" + "=" * 50 + "\n")

# ----------------------
# BLOQUE PRINCIPAL
# ----------------------
if __name__ == "__main__":
    texto = """
    El procesamiento del lenguaje natural es un campo fascinante de la inteligencia artificial.
    Los modelos de aprendizaje automático pueden procesar y entender el lenguaje humano.
    Las técnicas de aprendizaje profundo han revolucionado la forma en que procesamos datos de texto.
    Las bibliotecas de Python como spaCy y NLTK proporcionan una excelente funcionalidad.
    """

    print("TEXTO DE ENTRADA:")
    print("-" * 50)
    print(texto.strip())

    resultado = extraer_palabras_clave(texto, nlp)

    if resultado:
        mostrar_resultados(resultado)
    else:
        print("No se pudieron extraer las palabras clave.")
