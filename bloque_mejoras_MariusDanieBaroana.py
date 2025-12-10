"""
Mejoras del bloque original: logger y extracción de frases clave

Este módulo contiene dos mejoras importantes con respecto a
`bloque_origina_MariusDanielBaroana.py`:

1) `SessionLogger` ahora está construido sobre el módulo estándar
   `logging` con un `RotatingFileHandler`. Esto añade rotación de
   archivos, niveles de log, formato consistente y mejor compatibilidad
   con herramientas externas. (Mejora visible: logs más robustos y
   fácilmente configurables.)

2) `extraer_palabras_clave` mantiene las salidas anteriores pero añade
   `frases_clave`: extracción y puntuación de frases nominales (noun-chunks)
   multi-palabra usando spaCy. Las `frases_clave` suelen ser más
   informativas que palabras aisladas (ej.: "procesamiento del lenguaje natural").
   (Mejora visible: resultados más útiles en informes y presentaciones.)

El módulo incluye una demostración en el bloque `__main__` que muestra
los resultados y cómo se registran en el fichero de log.
"""

import os
import re
import unicodedata
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import Counter

# Dependencias NLP
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
    stopwords = None
    word_tokenize = None

# ----------------------
# Recursos compartidos (caché global)
# ----------------------
# Cargar stopwords de forma global para evitar costoso recálculo por llamada
_stopwords_es = None
if nltk and stopwords:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    _stopwords_es = set(stopwords.words('spanish'))
else:
    _stopwords_es = set()

# Intentar cargar spaCy si está disponible
_nlp = None
if spacy:
    try:
        _nlp = spacy.load('es_core_news_sm')
    except Exception:
        # Si el modelo no está instalado, no fallamos; dejamos _nlp en None
        _nlp = None

# ----------------------
# SessionLogger (mejora 1)
# ----------------------
class SessionLogger:
    """
    Logger de sesión basado en `logging` con rotación de archivos.

    Descripción extendida:
        Esta clase es una envoltura ligera sobre `logging.Logger` diseñada
        para reemplazar el logger manual del código original. Cada instancia
        crea un archivo de sesión en la carpeta `logs/` con nombre
        `session_YYYYMMDD_HHMMSS.log` y adjunta un `RotatingFileHandler`
        para controlar el tamaño del fichero y mantener historiales.

    Propósito y comportamiento:
        - Crear la carpeta `logs/` si no existe (seguridad con `exist_ok=True`).
        - Usar `RotatingFileHandler` para que los archivos no crezcan
          indefinidamente (`max_bytes` y `backup_count`).
        - Proveer métodos sencillos `log(...)` y `error(...)` para mantener
          una API similar al logger anterior y reducir cambios en llamadas.

    API pública:
        __init__(logs_dir='logs', max_bytes=5_000_000, backup_count=5):
            - logs_dir (str): directorio donde se almacenan los logs.
            - max_bytes (int): tamaño máximo del archivo antes de rotar.
            - backup_count (int): número de archivos antiguos a conservar.

        log(action: str, input_snippet: str, result):
            - Registra un evento en nivel INFO. `input_snippet` se recorta a
              120 caracteres para evitar filtrar datos sensibles o muy largos.
            - `result` se intenta serializar con `repr()`; si falla, se
              escribe un mensaje alternativo.

        error(message: str):
            - Registra un mensaje en nivel ERROR.

    Formato y extensibilidad:
        - Las entradas usan un `Formatter` con timestamp ISO y nivel, lo que
          facilita su parsing posterior y su visualización en herramientas.
        - Al usar la API de `logging`, es sencillo reenviar estos mensajes a
          otros handlers (stdout, syslog, Elastic, etc.) sin cambiar el
          código llamador.

    Ejemplo:
        >>> logger = SessionLogger()
        >>> logger.log('Extracción', 'fragmento...', {'top': [('palabra', 2)]})
    """

    def __init__(self, logs_dir='logs', max_bytes=5_000_000, backup_count=5):
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(logs_dir, f'session_{timestamp}.log')

        self.logger = logging.getLogger(f'session_{timestamp}')
        self.logger.setLevel(logging.INFO)

        # Evitar añadir múltiples handlers si se instancia varias veces
        if not self.logger.handlers:
            handler = RotatingFileHandler(filename, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.filename = filename

    def log(self, action: str, input_snippet: str, result):
        """Registra una entrada informativa con un snippet del input y el resultado.

        - Se recorta `input_snippet` a 120 caracteres para evitar registrar datos
          largos o sensibles accidentalmente.
        - `result` serializable a string se incluye en la línea de log.
        """
        snippet = (input_snippet or '')[:120]
        try:
            self.logger.info('%s | Entrada: %s | Resultado: %s', action, snippet, repr(result))
        except Exception:
            # Fallback sencillo si la serialización falla
            self.logger.info('%s | Entrada: %s | Resultado (no serializable)', action, snippet)

    def error(self, message: str):
        self.logger.error(message)


# Instancia simple para uso en el módulo
logger = SessionLogger()

# ----------------------
# Función: extraer_palabras_clave (mejora 2)
# ----------------------
def _normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación y tokenización.

    Detalles:
        - Normaliza la representación Unicode usando `NFKC` para reducir
          variaciones en la codificación que pueden afectar a la tokenización.
        - Aplica `casefold()` para una comparación insensible a mayúsculas
          y más robusta que `lower()` en contextos multilingües.

    Args:
        text (str): Texto de entrada.

    Returns:
        str: Texto normalizado; devuelve cadena vacía si la entrada no es `str`.
    """
    if not isinstance(text, str):
        return ''
    nfkd = unicodedata.normalize('NFKC', text)
    return nfkd.casefold()


def extraer_palabras_clave(texto: str, nlp=None, top_n=5, top_phrases=6):
    """
    Extrae palabras y frases clave de un texto en español.

    Proceso (paso a paso):
        1. Validación: si `texto` no es `str` o está vacío, devuelve `None`.
        2. Normalización: aplica `_normalize_text` para normalizar Unicode
           y casefold.
        3. Tokenización y conteo de tokens:
           - Si NLTK está disponible, se usa `word_tokenize` y el conjunto
             de `stopwords` cacheado (`_stopwords_es`).
           - Si NLTK no está disponible, se aplica una expresión regular
             Unicode para tokenizar palabras básicas.
        4. Extracción morfosintáctica (opcional):
           - Si se proporciona `nlp` (spaCy), se procesa `doc = nlp(texto)`.
           - Se extraen sustantivos y verbos mediante `token.pos_` y
             se cuentan por lema (`token.lemma_`) para agrupar formas.
        5. Frases clave (noun-chunks):
           - Si `nlp` está disponible, se recorren `doc.noun_chunks`, se
             filtran tokens vacíos o stopwords y se construyen frases por
             lema.
           - Se calcula un `score` simple que favorece frases de varias
             palabras y aquellas que contienen sustantivos frecuentes.

    Args:
        texto (str): Texto de entrada. Si está vacío o no es `str`, la
            función devolverá `None`.
        nlp (Optional[Language]): objeto spaCy (por ejemplo `es_core_news_sm`)
            para análisis morfosintáctico y extracción de noun-chunks. Si es
            `None`, la función hará un fallback y no generará `frases_clave`.
        top_n (int): número de elementos a devolver en las listas de palabras,
            sustantivos y verbos.
        top_phrases (int): número máximo de frases clave a devolver.

    Returns:
        dict | None: Diccionario con las claves:
            - 'top_5_palabras': list[tuple(str, int)]
            - 'sustantivos': list[tuple(str, int)]  # lemas de sustantivos
            - 'verbos': list[tuple(str, int)]       # lemas de verbos
            - 'frases_clave': list[tuple(str, float)]  # frase lematizada + score
        Devuelve `None` si la entrada no es válida.

    Notas:
        - Se cachean las stopwords a nivel de módulo para mejorar rendimiento
          en llamadas repetidas o procesamiento por lotes.
        - La función intenta ser tolerante: si spaCy o NLTK no están
          instalados, sigue devolviendo la mayor parte de la información
          posible usando fallback simple.
        - El score de `frases_clave` es intencionadamente simple y explicable
          (bueno para informes académicos); puede reemplazarse por TF-IDF
          o RAKE/YAKE para usos más avanzados.

    Ejemplo resumido:
        >>> extraer_palabras_clave('El procesamiento del lenguaje natural...', nlp=nlp)
        {
            'top_5_palabras': [('procesamiento', 1), ...],
            'sustantivos': [('procesamiento', 1), ...],
            'verbos': [('procesar', 1), ...],
            'frases_clave': [('procesamiento del lenguaje natural', 3.6), ...]
        }
    """
    if not texto or not isinstance(texto, str) or not texto.strip():
        return None

    texto_norm = _normalize_text(texto)

    # ------------------
    # Tokens y conteo (NLTK o fallback simple)
    # ------------------
    tokens_filtered = []
    if nltk and word_tokenize:
        tokens = word_tokenize(texto_norm)
        stopwords_es = _stopwords_es
        tokens_filtered = [t for t in tokens if re.fullmatch(r"[\w\-áéíóúñü]+", t, flags=re.UNICODE) and t not in stopwords_es and len(t) > 2]
        top_5 = Counter(tokens_filtered).most_common(top_n)
    else:
        # Fallback: tokenizar por palabra simple
        tokens = re.findall(r"[\w\-áéíóúñü]+", texto_norm, flags=re.UNICODE)
        tokens_filtered = [t for t in tokens if t not in _stopwords_es and len(t) > 2]
        top_5 = Counter(tokens_filtered).most_common(top_n)

    # ------------------
    # Sustantivos y verbos (usar spaCy si está disponible)
    # ------------------
    sustantivos_relevantes = []
    verbos_principales = []
    doc = None
    if nlp:
        try:
            doc = nlp(texto)
            # Usar lemas para agrupar formas flexionadas
            sustantivos_relevantes = Counter([token.lemma_.casefold() for token in doc if token.pos_ == 'NOUN' and len(token.lemma_) > 2]).most_common(top_n)
            verbos_principales = Counter([token.lemma_.casefold() for token in doc if token.pos_ == 'VERB' and len(token.lemma_) > 2]).most_common(top_n)
        except Exception as e:
            logger.error(f'Error procesando con spaCy: {e}')
            # Dejar listas vacías o caer al fallback

    # ------------------
    # Frases clave (noun-chunks) con puntuación simple
    # ------------------
    frases_clave = []
    if nlp and doc is not None:
        # Contar ocurrencias de noun_chunks (normalizados)
        phrase_counter = Counter()
        for chunk in doc.noun_chunks:
            # Normalizar y limpiar los bordes
            words = [t for t in chunk if not t.is_stop and t.is_alpha]
            if not words:
                continue
            # Construir una versión canónica basada en lemas
            phrase_lemmas = ' '.join([t.lemma_.casefold() for t in words])
            if len(phrase_lemmas) < 3:
                continue
            phrase_counter[phrase_lemmas] += 1

        # Preparar lista y aplicar puntuación que prioriza frases multi-palabra
        sustantivos_top = {s for s, _ in sustantivos_relevantes}
        scored_phrases = []
        for phrase, freq in phrase_counter.items():
            num_words = len(phrase.split())
            contains_top_noun = 1 if any(word in sustantivos_top for word in phrase.split()) else 0
            # Fórmula simple: frecuencia * (1 + 0.5 * (num_words - 1)) + 0.2 * contains_top_noun
            score = freq * (1 + 0.5 * (num_words - 1)) + 0.2 * contains_top_noun
            scored_phrases.append((phrase, round(score, 3)))

        frases_clave = sorted(scored_phrases, key=lambda x: x[1], reverse=True)[:top_phrases]

    resultado = {
        'top_5_palabras': top_5,
        'sustantivos': sustantivos_relevantes,
        'verbos': verbos_principales,
        'frases_clave': frases_clave,
    }

    # Registrar resultado (se registra un snippet del texto dentro del logger)
    logger.log('Extracción de Palabras Clave (mejorada)', texto[:120], resultado)

    return resultado


# ----------------------
# Función auxiliar para mostrar resultados en consola
# ----------------------
def mostrar_resultados(palabras_clave):
    """
    Muestra los resultados de la extracción de palabras clave en consola.

    Args:
        palabras_clave (dict): Diccionario devuelto por `extraer_palabras_clave`.
            Debe contener las claves 'top_5_palabras', 'sustantivos', 'verbos'
            y 'frases_clave'.

    Comportamiento:
        - Si `palabras_clave` es `None` o vacío imprime un mensaje de error
          y retorna.
        - Imprime secciones separadas para "TOP PALABRAS", "SUSTANTIVOS",
          "VERBOS" y "FRASES CLAVE" en un formato legible para terminal.

    Notas:
        - Esta función no modifica los datos; sólo formatea y muestra el
          contenido en la salida estándar. Está pensada para demostraciones
          y debugging, y no para serialización de resultados.
    """
    if not palabras_clave:
        print('No hay resultados para mostrar.')
        return

    print('\n' + '=' * 60)
    print('RESULTADOS DE EXTRACCIÓN (MEJORADO)')
    print('=' * 60)

    print('\nTOP PALABRAS:')
    for i, (palabra, freq) in enumerate(palabras_clave['top_5_palabras'], 1):
        print(f'  {i}. {palabra}: {freq} veces')

    print('\nSUSTANTIVOS:')
    for i, (sustantivo, freq) in enumerate(palabras_clave['sustantivos'], 1):
        print(f'  {i}. {sustantivo}: {freq} veces')

    print('\nVERBOS:')
    for i, (verbo, freq) in enumerate(palabras_clave['verbos'], 1):
        print(f'  {i}. {verbo}: {freq} veces')

    print('\nFRASES CLAVE:')
    if palabras_clave['frases_clave']:
        for i, (frase, score) in enumerate(palabras_clave['frases_clave'], 1):
            print(f'  {i}. {frase}: score={score}')
    else:
        print('  (spaCy no disponible o no se encontraron frases)')

    print('\n' + '=' * 60 + '\n')


# ----------------------
# DEMO: bloque principal
# ----------------------
if __name__ == '__main__':
    sample_text = """
    El procesamiento del lenguaje natural es un campo fascinante de la inteligencia artificial.
    Los modelos de aprendizaje automático pueden procesar y entender el lenguaje humano.
    Las técnicas de aprendizaje profundo han revolucionado la forma en que procesamos datos de texto.
    Las bibliotecas de Python como spaCy y NLTK proporcionan una excelente funcionalidad.
    """

    print('Demostración del módulo "bloque_mejoras"')
    print('Logs -->', logger.filename)

    resultado = extraer_palabras_clave(sample_text, nlp=_nlp)
    mostrar_resultados(resultado)

    print('Demo finalizada.')
