"""cli.py

Interfaz de línea de comandos para wordChef: integra todas las utilidades.

Este módulo contiene:
- Menú principal interactivo
- Sistema de logging automático (registra cada análisis en un archivo log)
- Integración de todas las funciones de análisis de texto
"""

import os
from datetime import datetime

# ============================================================================
# IMPORTACIONES - INTENTA CARGAR DE MÓDULOS SEPARADOS
# ============================================================================
# Si los módulos están en archivos separados, importar de ahí
# Si no están disponibles, los valores serán None y se mostrarán avisos
try:
	from loader import cargar_modelo_spacy
except ImportError:
	cargar_modelo_spacy = None

try:
	from normalizer import normalizador_texto
except ImportError:
	normalizador_texto = None

try:
	from patterns import encontrar_fechas, encontrar_dinero, encontrar_correos
except ImportError:
	encontrar_fechas = encontrar_dinero = encontrar_correos = None

try:
	from summarizer import resumen_simple
except ImportError:
	resumen_simple = None

try:
	from ner import extraer_entidades
except ImportError:
	extraer_entidades = None

try:
	from keywords import extraer_palabras_clave
except ImportError:
	extraer_palabras_clave = None

try:
	from sentiment import sentimiento_es
except ImportError:
	sentimiento_es = None


# ============================================================================
# SISTEMA DE LOGGING DE SESIÓN
# ============================================================================
# Este sistema registra automáticamente todas las análisis en un archivo log
# Cada sesión (ejecución del programa) crea un archivo nuevo con timestamp único

class SessionLogger:
	"""
	Clase para registrar análisis en un archivo de log por sesión.
	
	Cada vez que ejecutas el programa, se crea un archivo nuevo llamado:
	logs/session_YYYYMMDD_HHMMSS.log
	
	Todas las análisis se guardan automáticamente en este archivo.
	"""
	
	def __init__(self):
		"""Inicializa el logger y crea el archivo de sesión."""
		# Crear directorio de logs si no existe
		# os.path.exists() comprueba si la carpeta ya existe
		# os.makedirs() crea la carpeta (y carpetas padre si es necesario)
		if not os.path.exists("logs"):
			os.makedirs("logs")
			print("📁 Carpeta 'logs' creada automáticamente.")
		
		# Generar timestamp único para esta sesión (año-mes-día_hora-minuto-segundo)
		# Ejemplo: 20251126_161518 = 26/11/2025 a las 16:15:18
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# Crear nombre del archivo: logs/session_[timestamp].log
		self.filename = f"logs/session_{timestamp}.log"
		
		# Llamar función para escribir encabezado al inicio del archivo
		self._escribir_encabezado()
	
	def _escribir_encabezado(self):
		"""
		Escribe el encabezado del archivo de log.
		Indica fecha/hora y título de la sesión.
		"""
		# Abrir archivo en modo escritura ('w') - crea nuevo si no existe
		# encoding='utf-8' permite caracteres especiales español (ñ, tildes, etc.)
		with open(self.filename, 'w', encoding='utf-8') as f:
			# Escribir línea decorativa
			f.write("="*80 + "\n")
			# Escribir título con fecha y hora actual
			f.write(f"  SESIÓN wordChef - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			# Otra línea decorativa
			f.write("="*80 + "\n\n")
	
	def registrar(self, tipo_analisis, entrada, resultado):
		"""
		Registra un análisis completo en el archivo log.
		
		Parámetros:
		- tipo_analisis: nombre del análisis (ej: "Palabras clave", "Normalizador")
		- entrada: texto que se analizó (primeros 100 caracteres)
		- resultado: resultado del análisis (dict, string, list, etc.)
		"""
		# Abrir archivo en modo append ('a') - añade al final sin borrar
		with open(self.filename, 'a', encoding='utf-8') as f:
			# Escribir timestamp (hora:minuto:segundo) y tipo de análisis
			f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {tipo_analisis}\n")
			
			# Línea separadora para legibilidad
			f.write("-"*80 + "\n")
			
			# Mostrar primeros 100 caracteres de la entrada (o toda si es más corta)
			# Si es más larga, añadir "..." al final
			entrada_truncada = entrada[:100] + ('...' if len(entrada) > 100 else '')
			f.write(f"Entrada: {entrada_truncada}\n\n")
			
			# Formatear el resultado según su tipo
			if isinstance(resultado, dict):  # Si es un diccionario
				# Recorrer cada clave-valor del diccionario
				for clave, valor in resultado.items():
					f.write(f"  {clave}:\n")  # Escribir nombre de la clave
					
					# Si el valor es lista o conjunto, mostrar cada elemento con viñeta
					if isinstance(valor, (list, set)):
						for item in valor:
							f.write(f"    • {item}\n")  # Viñeta (bullet point)
					else:
						# Si es otro tipo (string, número, etc.), mostrar directamente
						f.write(f"    {valor}\n")
			else:
				# Si no es diccionario, mostrar resultado tal cual
				f.write(f"Resultado: {resultado}\n")
			
			# Separador final
			f.write("\n" + "="*80 + "\n\n")
	
	def registrar_patron(self, tipo_patron, coincidencias):
		"""
		Registra hallazgos de patrones (fechas, dinero, correos, etc.).
		
		Parámetros:
		- tipo_patron: qué tipo de patrón se buscó (ej: "Fechas", "Dinero")
		- coincidencias: lista de coincidencias encontradas (puede estar vacía)
		"""
		# Abrir archivo en modo append para añadir registro
		with open(self.filename, 'a', encoding='utf-8') as f:
			# Timestamp y tipo de búsqueda
			f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Búsqueda de patrones\n")
			
			# Línea separadora
			f.write("-"*80 + "\n")
			
			# Mostrar tipo de patrón y coincidencias (o "Ninguno encontrado" si está vacío)
			resultado = coincidencias if coincidencias else 'Ninguno encontrado'
			f.write(f"  {tipo_patron}: {resultado}\n")
			
			# Separador final
			f.write("\n" + "="*80 + "\n\n")

# Crear instancia global del logger
# Esto se ejecuta una sola vez al inicio del programa
# Después, todas las funciones pueden usar 'logger' para registrar cosas
try:
	logger = SessionLogger()
	print(f"📝 Sesión iniciada. Logs guardados en: {logger.filename}\n")
except Exception as e:
	print(f"⚠️  Aviso: No se pudo crear el logger: {e}")
	logger = None


# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================

def menu_principal():
	"""
	Menú interactivo principal.
	Permite al usuario seleccionar qué análisis desea realizar.
	"""
	
	# Cargar modelo de spaCy (puede ser None si no está disponible)
	if cargar_modelo_spacy is not None:
		nlp = cargar_modelo_spacy()
	else:
		nlp = None
		print("⚠️  Aviso: Función cargar_modelo_spacy no disponible.")

	while True:
		print("\n=== wordChef: Menú de utilidades ===")
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

		# ========== OPCIÓN 1: NORMALIZADOR ==========
		if opcion == '1':
			if normalizador_texto is None:
				print("❌ Error: Función normalizador_texto no disponible.")
				continue
			
			texto = input("Ingresa texto a normalizar:\n> ")
			res = normalizador_texto(texto, nlp)
			if res is None:
				print("Texto inválido.")
			else:
				print("\n--- RESULTADOS ---")
				print("Texto Original:\n", res['original'])
				print("\nTexto Lematizado:\n", res['lematizado'])
				print("\nTexto sin Repeticiones:\n", res['sin_repeticiones'])
				print("\nTexto Corregido:\n", res['corregido'])
				# Registrar en log
				if logger:
					logger.registrar("Normalizador", texto, res)

		# ========== OPCIÓN 2: PATRONES ==========
		elif opcion == '2':
			if encontrar_fechas is None:
				print("❌ Error: Funciones de patrones no disponibles.")
				continue
			
			texto = input("Introduce texto para analizar patrones:\n")
			fechas = encontrar_fechas(texto)
			dinero = encontrar_dinero(texto)
			correos = encontrar_correos(texto)
			print('\nFechas encontradas:', fechas or 'Ninguna')
			print('Cifras de dinero:', dinero or 'Ninguna')
			print('Correos electrónicos:', correos or 'Ninguno')
			# Registrar en log
			if logger:
				logger.registrar_patron("Fechas", fechas)
				logger.registrar_patron("Dinero", dinero)
				logger.registrar_patron("Correos", correos)

		# ========== OPCIÓN 3: RESUMEN ==========
		elif opcion == '3':
			if resumen_simple is None:
				print("❌ Error: Función resumen_simple no disponible.")
				continue
			
			texto = input("Introduce texto para resumir:\n")
			resumen = resumen_simple(texto, n=3, nlp=nlp)
			print('\n--- RESUMEN ---')
			print(resumen)
			# Registrar en log
			if logger:
				logger.registrar("Resumen simple", texto, f"Resumen: {resumen}")

		# ========== OPCIÓN 4: EXTRACCIÓN NER ==========
		elif opcion == '4':
			if extraer_entidades is None:
				print("❌ Error: Función extraer_entidades no disponible.")
				continue
			
			texto = input("Introduce texto para extraer entidades (NER):\n")
			entidades = extraer_entidades(texto, nlp)
			for k, v in entidades.items():
				print(f"{k}: {v if v else 'Ninguno detectado'}")
			# Registrar en log
			if logger:
				logger.registrar("Extracción NER", texto, entidades)

		# ========== OPCIÓN 5: PALABRAS CLAVE ==========
		elif opcion == '5':
			if extraer_palabras_clave is None:
				print("❌ Error: Función extraer_palabras_clave no disponible.")
				continue
			
			texto = input("Introduce texto para extraer palabras clave:\n")
			resultado = extraer_palabras_clave(texto, nlp=nlp)
			if not resultado:
				print("No se pudieron extraer palabras clave.")
			else:
				print('\nTop 5 palabras:', resultado['top_5_palabras'])
				print('Sustantivos relevantes:', resultado['sustantivos'])
				print('Verbos principales:', resultado['verbos'])
				# Registrar en log
				if logger:
					logger.registrar("Palabras clave", texto, resultado)

		# ========== OPCIÓN 6: SENTIMIENTO ==========
		elif opcion == '6':
			if sentimiento_es is None:
				print("❌ Error: Función sentimiento_es no disponible.")
				continue
			
			texto = input("Introduce texto para análisis de sentimiento:\n")
			sentimiento, score, raw = sentimiento_es(texto)
			if sentimiento == "Error":
				print(f"Ocurrió un error: {raw}")
			else:
				print(f"Resultado: {sentimiento} (Confianza: {score:.4f}) Etiqueta: {raw}")
				# Registrar en log
				if logger:
					logger.registrar("Análisis de sentimiento", texto, 
						{"Sentimiento": sentimiento, "Confianza": f"{score:.4f}", "Etiqueta": raw})

		# ========== OPCIÓN INVÁLIDA ==========
		else:
			print("Opción no válida, intenta de nuevo.")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
	try:
		menu_principal()
	except KeyboardInterrupt:
		print("\n\nPrograma interrumpido por el usuario.")
	except Exception as e:
		print(f"\n❌ Error inesperado: {e}")
