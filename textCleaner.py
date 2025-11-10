import re
import os
from pathlib import Path
from collections import Counter, defaultdict

# --- CONFIGURACIÓN PRINCIPAL ---

# 1. Carpeta con los .txt originales (los que quieres limpiar)
CARPETA_ENTRADA = Path("H:/Mi unidad/Classroom/26PIA/Proyecto RAG/texto/txt")

# 2. Carpeta NUEVA donde se guardarán los .txt limpios
CARPETA_SALIDA = Path("H:/Mi unidad/Classroom/26PIA/Proyecto RAG/texto/txtLimpieza")

# --- AJUSTES DE HEURÍSTICA (Reglas de limpieza) ---

# Regla 2: Eliminación de Texto Repetitivo
# Si una línea aparece en más de este porcentaje de documentos, se considera ruido.
# (Ej. 0.3 = 30% de los documentos)
PORCENTAJE_MINIMO_RUIDO = 0.3

# Para la Regla 2, no analizar líneas que sean más cortas que esto
# (evita que "a", "y", "de" se marquen como ruido si están solos)
LONGITUD_MINIMA_LINEA_RUIDO = 10 

# Regla 3: Eliminación de Ruido Estructural
# Líneas con menos de esta cantidad de palabras serán eliminadas
# (siempre y cuando no sean un título o frase que termine con puntuación)
MIN_PALABRAS_LINEA = 3

# Puntuación válida que "salva" a una línea corta de ser eliminada
PUNTUACION_FINAL_VALIDA = ('.', ',', ':', ';', '!', '?', ')', ']', '"', "'")

# --- FIN DE LA CONFIGURACIÓN ---


def es_ruido(linea, ruido_repetitivo_set):
    """
    Comprueba una sola línea contra todas nuestras reglas de ruido.
    Devuelve True si la línea es ruido y debe ser eliminada.
    """
    linea_limpia = linea.strip()
    linea_norm = linea_limpia.lower()

    # Regla 0: Ignorar líneas completamente vacías (no son ruido, son formato)
    if not linea_limpia:
        return False

    # Regla 1: Eliminación de Índices (TOC)
    # Patrón: "Cualquier texto" ... (5+ puntos o guiones) ... "un número"
    if re.search(r".*[.-]{5,}\s*\d+\s*$", linea_limpia):
        return True

    # Regla 2: Eliminación de Texto Repetitivo (Cabeceras/Pies de página)
    # Comprueba si la línea (normalizada) está en nuestro set de ruido
    if linea_norm in ruido_repetitivo_set:
        return True

    # Regla 3.1: Ruido Estructural (Números de página)
    # Patrón: "Solo un número en la línea"
    if re.match(r"^\s*\d+\s*$", linea_limpia):
        return True

    # Regla 3.2: Ruido Estructural (Líneas cortas)
    palabras = linea_limpia.split()
    if len(palabras) > 0 and len(palabras) < MIN_PALABRAS_LINEA:
        # La línea es corta. ¿Termina con puntuación?
        if not linea_limpia.endswith(PUNTUACION_FINAL_VALIDA):
            # Es corta y no parece una frase. Es ruido.
            return True

    # Si pasó todas las pruebas, no es ruido
    return False

def limpiar_archivos_txt():
    """
    Función principal que orquesta el análisis y la limpieza.
    """
    
    # Asegurarse de que la carpeta de salida exista
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    
    archivos_txt = list(CARPETA_ENTRADA.glob("*.txt"))
    if not archivos_txt:
        print(f"Error: No se encontraron archivos .txt en {CARPETA_ENTRADA}")
        return

    num_documentos = len(archivos_txt)
    print(f"Encontrados {num_documentos} archivos para procesar.")

    # --- FASE 1: ANÁLISIS (Construir base de datos de ruido) ---
    print("\n--- FASE 1: Analizando todos los archivos para detectar ruido... ---")
    
    # Usamos un Counter para la frecuencia total
    contador_frecuencia_lineas = Counter()
    # Usamos un defaultdict para saber en qué archivos aparece cada línea
    mapa_linea_a_archivos = defaultdict(set)

    for ruta_txt in archivos_txt:
        try:
            with open(ruta_txt, "r", encoding="utf-8") as f:
                for linea in f:
                    linea_norm = linea.strip().lower()
                    
                    # Solo analizamos líneas de una longitud razonable
                    if len(linea_norm) > LONGITUD_MINIMA_LINEA_RUIDO:
                        contador_frecuencia_lineas[linea_norm] += 1
                        mapa_linea_a_archivos[linea_norm].add(ruta_txt.name)
        except Exception as e:
            print(f"  Error leyendo {ruta_txt.name}: {e}")

    # Ahora, construimos el "set" de ruido repetitivo
    ruido_repetitivo_set = set()
    umbral_documentos = num_documentos * PORCENTAJE_MINIMO_RUIDO
    
    for linea, archivos in mapa_linea_a_archivos.items():
        # Si la línea aparece en más del X% de los documentos...
        if len(archivos) > umbral_documentos:
            ruido_repetitivo_set.add(linea)

    print(f"Análisis completado. Se identificaron {len(ruido_repetitivo_set)} líneas de ruido repetitivo.")


    # --- FASE 2: LIMPIEZA (Escribir archivos limpios) ---
    print("\n--- FASE 2: Limpiando y escribiendo nuevos archivos... ---")

    for ruta_txt in archivos_txt:
        nombre_archivo = ruta_txt.name
        print(f"  Procesando: {nombre_archivo}")
        
        lineas_limpias = []
        try:
            # Leemos el archivo original
            with open(ruta_txt, "r", encoding="utf-8") as f_in:
                for linea in f_in:
                    # Aplicamos todas las reglas
                    if not es_ruido(linea, ruido_repetitivo_set):
                        lineas_limpias.append(linea)
            
            # Escribimos el archivo limpio en la carpeta de salida
            ruta_salida_fichero = CARPETA_SALIDA / nombre_archivo
            with open(ruta_salida_fichero, "w", encoding="utf-8") as f_out:
                f_out.writelines(lineas_limpias)
                
        except Exception as e:
            print(f"  Error procesando {nombre_archivo}: {e}")

    print("\n¡Proceso de limpieza completado!")

    # --- FASE 3: INFORME (Log) ---
    print("\n--- INFORME DE RUIDO ---")
    
    if not ruido_repetitivo_set:
        print("No se identificó ruido repetitivo común (según el umbral).")
    else:
        print(f"Se eliminaron las siguientes {len(ruido_repetitivo_set)} líneas repetitivas:")
        for i, linea_ruido in enumerate(ruido_repetitivo_set):
            if i > 20: # Mostramos solo las primeras 20 para no saturar
                print(f"  ... y {len(ruido_repetitivo_set) - i} más.")
                break
            print(f"  - \"{linea_ruido[:80]}...\"") # Mostramos un fragmento

    print("\nTop 10 líneas más frecuentes en general (para depuración):")
    for linea, frecuencia in contador_frecuencia_lineas.most_common(10):
        print(f"  (Freq: {frecuencia}) \"{linea[:80]}...\"")


# --- Ejecutar la función ---
if __name__ == "__main__":
    limpiar_archivos_txt()