import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# --- PARÁMETROS CONFIGURABLES ---

# 1. Carpeta con los .txt limpios
CARPETA_ENTRADA = Path("H:/Mi unidad/Classroom/26PIA/Proyecto RAG/texto/txtLimpieza")

# 2. Archivo de salida (se creará si no existe)
CARPETA_SALIDA = CARPETA_ENTRADA.parent.parent / "chunks_individuales"

# 3. Tamaño deseado del chunk (en caracteres)
CHUNK_SIZE_OBJETIVO = 1500

# 4. Solapamiento entre chunks (en caracteres)
CHUNK_OVERLAP = 200

# --- ESTRATEGIA DE PARTICIÓN ---

# Paso 1: Partición Estructural (Regex para identificar cabeceras)
# Añade aquí tus patrones. ¡El uso de (paréntesis) es importante!
# Captura el separador para usarlo como metadato.
SEPARADORES_LOGICOS_REGEX = [
    r"(\nCapítulo \d+.*?)\n",
    r"(\nArtículo \d+.*?)\n",
    r"(\nSección \d+.*?)\n",
    r"(\nApartado [A-Z].*?)\n",
    r"(\nEpígrafe \d+\.\d+.*?)\n"
]

# Paso 2: Partición Recursiva (Separadores semánticos, en orden)
SEPARADORES_RECURSIVOS = ["\n\n", "\n", ". ", " ", ""]

# --- LÓGICA DEL SCRIPT ---

def _particionar_recursivo(texto: str, size: int, separadores: List[str]) -> List[str]:
    """
    Divide un texto de forma recursiva usando los separadores,
    del más general al más específico.
    """
    if len(texto) <= size:
        return [texto]

    # Elige el separador más relevante de la lista
    separador_actual = separadores[0]
    resto_separadores = separadores[1:]

    # Si no podemos dividir más (llegamos al separador de "caracteres")
    if separador_actual == "":
        return [texto[i:i + size] for i in range(0, len(texto), size)]

    # Intenta dividir por el separador actual
    chunks_divididos = []
    try:
        # Usamos re.split para manejar patrones más complejos (como ". ")
        divisiones = re.split(f"({re.escape(separador_actual)})", texto)
        
        # 'divisiones' será [texto, sep, texto, sep, texto...]
        # Reagrupamos para mantener el separador al final de cada chunk
        partes = []
        for i in range(0, len(divisiones), 2):
            parte = divisiones[i]
            if i + 1 < len(divisiones):
                parte += divisiones[i+1]
            if parte:
                partes.append(parte)

        if not partes:
            partes = [texto] # Fallback por si la división falla
        
    except re.error:
        # Fallback si el separador es simple (ej. \n\n)
        partes = texto.split(separador_actual)

    # Ahora, procesa las partes
    for parte in partes:
        if len(parte) > size:
            # Esta parte sigue siendo muy grande, llamamos recursivamente
            # con el *siguiente* nivel de separadores
            chunks_divididos.extend(
                _particionar_recursivo(parte, size, resto_separadores)
            )
        else:
            chunks_divididos.append(parte)
    
    return chunks_divididos

def _merge_chunks(chunks_naturales: List[str], size: int, overlap: int) -> List[str]:
    """
    Toma una lista de chunks "naturales" (párrafos, frases)
    y los agrupa en chunks de tamaño 'size' con solapamiento 'overlap'.
    """
    if not chunks_naturales:
        return []

    chunks_finales = []
    chunk_actual = ""
    
    # Usamos el primer separador (ej. "\n\n") como "pegamento"
    pegamento = SEPARADORES_RECURSIVOS[0] 

    for chunk in chunks_naturales:
        # Si añadir el siguiente chunk (con pegamento) supera el tamaño...
        if len(chunk_actual) + len(chunk) + len(pegamento) > size:
            if chunk_actual:
                chunks_finales.append(chunk_actual.strip())
            
            # Empezamos el nuevo chunk, pero con solapamiento
            # El solapamiento es el final del chunk que acabamos de guardar
            texto_overlap = chunk_actual[-overlap:]
            chunk_actual = texto_overlap + pegamento + chunk
        else:
            # Seguimos construyendo el chunk actual
            if chunk_actual:
                chunk_actual += pegamento + chunk
            else:
                chunk_actual = chunk # Es el primer chunk
    
    # Añadir el último chunk
    if chunk_actual:
        chunks_finales.append(chunk_actual.strip())
    
    return chunks_finales

def procesar_documento(
    texto: str, 
    nombre_archivo: str, 
    chunk_id_global: int
) -> List[Dict[str, Any]]:
    """
    Procesa un único documento de texto y lo convierte en una lista
    de objetos JSON (chunks).
    """
    
    # --- Paso 1: Partición Estructural (Lógica) ---
    master_regex = "|".join(SEPARADORES_LOGICOS_REGEX)
    bloques_con_separadores = re.split(f"({master_regex})", texto, flags=re.IGNORECASE)
    
    bloques_logicos = []
    seccion_actual = "General" # Sección por defecto
    
    for i, parte in enumerate(bloques_con_separadores):
        if not parte:
            continue
            
        # Si 'parte' es un separador (índice impar tras re.split con captura)
        if i % 2 == 1:
            seccion_actual = parte.strip()
        else:
            # Es texto normal
            bloques_logicos.append((parte, seccion_actual))

    # --- Paso 2: Partición Recursiva y Fusión ---
    lista_json_chunks = []
    
    for bloque, seccion in bloques_logicos:
        if not bloque.strip():
            continue
            
        # Dividimos semánticamente el bloque
        chunks_naturales = _particionar_recursivo(
            bloque, CHUNK_SIZE_OBJETIVO, SEPARADORES_RECURSIVOS
        )
        
        # Agrupamos los chunks naturales en trozos finales con solapamiento
        chunks_fusionados = _merge_chunks(
            chunks_naturales, CHUNK_SIZE_OBJETIVO, CHUNK_OVERLAP
        )
        
        # Creamos los objetos JSON
        for i, contenido_chunk in enumerate(chunks_fusionados):
            chunk_id = f"{nombre_archivo.replace('.txt', '')}_chunk_{chunk_id_global:05d}"
            
            # Opcional: Inferir URL (dejamos como placeholder)
            url_placeholder = f"https://placeholder.com/{nombre_archivo}"
            
            datos_chunk = {
                "content": contenido_chunk,
                "metadata": {
                    "source_file": nombre_archivo,
                    "logical_section": seccion,
                    "chunk_id": chunk_id,
                    "url_source": url_placeholder 
                }
            }
            lista_json_chunks.append(datos_chunk)
            chunk_id_global += 1
            
    return lista_json_chunks, chunk_id_global

def main():
    """
    Función principal que orquesta todo el proceso.
    """
    if not CARPETA_ENTRADA.exists():
        print(f"Error: La carpeta de entrada no existe: {CARPETA_ENTRADA}")
        return

    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    total_chunks_generados = 0
    
    print(f"Iniciando el proceso de particionado...")
    print(f"Fuente: {CARPETA_ENTRADA}")
    print(f"Salida: {CARPETA_SALIDA}")
    print(f"Tamaño/Overlap: {CHUNK_SIZE_OBJETIVO}/{CHUNK_OVERLAP}\n")
    
    # Abrimos el archivo de salida en modo escritura
    archivos_txt = list(CARPETA_ENTRADA.glob("*.txt"))
    
    if not archivos_txt:
        print("Advertencia: No se encontraron archivos .txt en la carpeta de entrada.")
        return

    # --- MODIFICADO: No hay 'with open' global ---
    for ruta_txt in archivos_txt:
        nombre_archivo = ruta_txt.name
        print(f"Procesando: {nombre_archivo}...")
        
        # --- NUEVO: Definir ruta de salida para este archivo ---
        nombre_base = ruta_txt.stem
        nombre_salida = f"chunk_{nombre_base}.jsonl"
        ruta_salida = CARPETA_SALIDA / nombre_salida
        
        try:
            with open(ruta_txt, "r", encoding="utf-8") as f_in:
                texto = f_in.read()
            
            # Procesamos el documento, reseteando el contador a 0
            lista_json, _ = procesar_documento(
                texto, nombre_archivo, 0 
            )
            
            # --- NUEVO: Escribir en el archivo de salida individual ---
            with open(ruta_salida, "w", encoding="utf-8") as f_out:
                for chunk_json in lista_json:
                    # json.dumps con ensure_ascii=False para tildes
                    f_out.write(json.dumps(chunk_json, ensure_ascii=False) + "\n")
            
            print(f"  -> Generados {len(lista_json)} chunks. -> {ruta_salida}")
            total_chunks_generados += len(lista_json)

        except Exception as e:
            print(f"  ❌ Error procesando {nombre_archivo}: {e}")

    print(f"\n¡Proceso completado!")
    print(f"Se generaron un total de {total_chunks_generados} chunks.")
    print(f"Resultados guardados en: {CARPETA_SALIDA}")

# --- Ejecutar el script ---
if __name__ == "__main__":
    main()