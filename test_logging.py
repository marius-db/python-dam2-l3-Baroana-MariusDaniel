"""
Test script to demonstrate the logging functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from wordChef import logger, encontrar_fechas, encontrar_dinero, encontrar_correos

# Test logging
texto_prueba = "Nací el 15/03/1995 y mi email es juan@ejemplo.com. Gané 1500€."

# Test pattern finding with logging
fechas = encontrar_fechas(texto_prueba)
dinero = encontrar_dinero(texto_prueba)
correos = encontrar_correos(texto_prueba)

logger.registrar_patron("Fechas", fechas)
logger.registrar_patron("Dinero", dinero)
logger.registrar_patron("Correos", correos)

print("✅ Logging completado!")
print(f"📝 Log guardado en: {logger.filename}")
