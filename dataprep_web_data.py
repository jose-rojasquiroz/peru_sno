"""
prepare_web_data.py - VERSIÓN CORREGIDA

Script para procesar los datos del análisis de orientación de calles
y generar los archivos necesarios para la aplicación web estática.
"""

import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar OSMNx
import osmnx as ox

# Para barras de progreso
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Instala tqdm para barras de progreso: pip install tqdm")

# Configurar paths - usando Path relativo al script
BASE_DIR = Path(__file__).parent  # Directorio donde está este script
DATA_DIR = BASE_DIR / "data"      # Carpeta data en la misma ubicación
WEB_DIR = BASE_DIR / "web"
WEB_DATA_DIR = WEB_DIR / "data"
WEB_GRAFICOS_DIR = WEB_DATA_DIR / "graficos"

# Crear directorios si no existen
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEB_GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

def procesar_poligonos_urbanos():
    """Cargar y procesar el archivo de polígonos urbanos"""
    print("Cargando polígonos urbanos...")
    
    # Definir rutas posibles para el archivo GPKG
    rutas_posibles = [
        DATA_DIR / "pu-principalesciudades-WGS84.gpkg",
        BASE_DIR / "data" / "pu-principalesciudades-WGS84.gpkg",
        BASE_DIR.parent / "data" / "pu-principalesciudades-WGS84.gpkg",
        Path("data/pu-principalesciudades-WGS84.gpkg"),
        Path("../data/pu-principalesciudades-WGS84.gpkg")
    ]
    
    gdf = None
    archivo_encontrado = None
    
    # Intentar cada ruta posible
    for ruta in rutas_posibles:
        if ruta.exists():
            print(f"  Encontrado: {ruta}")
            archivo_encontrado = ruta
            try:
                gdf = gpd.read_file(ruta)
                break
            except Exception as e:
                print(f"  Error leyendo {ruta}: {e}")
                continue
    
    if gdf is None:
        print("\n❌ ERROR: No se pudo encontrar el archivo pu-principalesciudades-WGS84.gpkg")
        print("\nBusqué en las siguientes ubicaciones:")
        for ruta in rutas_posibles:
            print(f"  • {ruta.absolute()}")
        print("\nPor favor, asegúrate de que el archivo existe en una de estas ubicaciones.")
        return None
    
    print(f"  Archivo cargado: {archivo_encontrado}")
    
    # Normalizar nombres de columnas
    gdf.columns = [col.lower() for col in gdf.columns]
    
    print(f"\n  Columnas disponibles: {list(gdf.columns)}")
    print(f"  Número de filas: {len(gdf)}")
    print(f"  Sistema de coordenadas: {gdf.crs}")
    
    # Mostrar primeras filas para diagnóstico
    print(f"\n  Primeras 3 ciudades:")
    for i, row in gdf.head(3).iterrows():
        ciudad_nombre = row.get('ciudad', 'N/A')
        poblacion = row.get('pob17', 'N/A')
        region = row.get('regnat', 'N/A')
        print(f"    {i}: {ciudad_nombre} - Pob: {poblacion} - Región: {region}")
    
    # Extraer departamento y provincia si están en el nombre de ciudad
    if 'departamento' not in gdf.columns and 'ciudad' in gdf.columns:
        print("\n  Extrayendo departamento y provincia del campo 'ciudad'...")
        
        def extract_info(nombre_ciudad):
            """Extraer ciudad, provincia y departamento del nombre completo"""
            if not isinstance(nombre_ciudad, str):
                return '', '', ''
            
            # Limpiar el nombre
            nombre = nombre_ciudad.strip()
            
            # Buscar patrones comunes
            if ',' in nombre:
                parts = [p.strip() for p in nombre.split(',')]
                
                # Patrón más común: "Ciudad, Departamento"
                if len(parts) == 2:
                    ciudad = parts[0]
                    departamento = parts[1]
                    # Intentar extraer provincia del departamento
                    provincia_parts = departamento.split()
                    provincia = provincia_parts[0] if len(provincia_parts) > 1 else departamento
                    return ciudad, provincia, departamento
                
                # Patrón: "Ciudad, Provincia, Departamento"
                elif len(parts) >= 3:
                    ciudad = parts[0]
                    provincia = parts[1]
                    departamento = parts[2]
                    return ciudad, provincia, departamento
            
            # Si no hay comas, asumir que es solo el nombre de la ciudad
            return nombre, '', ''
        
        # Aplicar la extracción
        extractions = gdf['ciudad'].apply(extract_info)
        gdf[['ciudad_nombre', 'provincia', 'departamento']] = pd.DataFrame(
            extractions.tolist(), 
            index=gdf.index
        )
        
        print(f"  Ejemplo de extracción:")
        for i in range(min(3, len(gdf))):
            print(f"    '{gdf.iloc[i]['ciudad']}' → Ciudad: '{gdf.iloc[i]['ciudad_nombre']}', Depto: '{gdf.iloc[i]['departamento']}'")
    else:
        gdf['ciudad_nombre'] = gdf['ciudad']
    
    # Si no se pudo extraer departamento, crear columna vacía
    if 'departamento' not in gdf.columns:
        gdf['departamento'] = ''
        print("  NOTA: Columna 'departamento' no encontrada, usando valor vacío")
    
    if 'provincia' not in gdf.columns:
        gdf['provincia'] = ''
        print("  NOTA: Columna 'provincia' no encontrada, usando valor vacío")
    
    # Renombrar columnas para consistencia
    rename_map = {}
    if 'pob17' in gdf.columns and 'poblacion' not in gdf.columns:
        rename_map['pob17'] = 'poblacion'
    if 'regnat' in gdf.columns and 'region_natural' not in gdf.columns:
        rename_map['regnat'] = 'region_natural'
    
    if rename_map:
        gdf.rename(columns=rename_map, inplace=True)
        print(f"  Columnas renombradas: {rename_map}")
    
    print(f"\n  Columnas finales: {list(gdf.columns)}")
    
    if 'region_natural' in gdf.columns:
        regiones = gdf['region_natural'].dropna().unique()
        print(f"  Regiones encontradas ({len(regiones)}): {regiones.tolist()}")
    elif 'regnat' in gdf.columns:
        regiones = gdf['regnat'].dropna().unique()
        print(f"  Regiones encontradas ({len(regiones)}): {regiones.tolist()}")
    
    return gdf

def generar_id_ciudad(nombre):
    """Generar ID único para la ciudad"""
    import re
    if not isinstance(nombre, str):
        return "ciudad_desconocida"
    
    # Limpiar nombre para ID
    id_ciudad = nombre.lower()
    # Reemplazar acentos
    id_ciudad = re.sub(r'[áàäâ]', 'a', id_ciudad)
    id_ciudad = re.sub(r'[éèëê]', 'e', id_ciudad)
    id_ciudad = re.sub(r'[íìïî]', 'i', id_ciudad)
    id_ciudad = re.sub(r'[óòöô]', 'o', id_ciudad)
    id_ciudad = re.sub(r'[úùüû]', 'u', id_ciudad)
    id_ciudad = re.sub(r'[ñ]', 'n', id_ciudad)
    # Reemplazar otros caracteres
    id_ciudad = re.sub(r'[^a-z0-9]', '_', id_ciudad)
    id_ciudad = re.sub(r'_+', '_', id_ciudad).strip('_')
    
    if id_ciudad == '':
        id_ciudad = 'ciudad_sin_nombre'
    
    return id_ciudad

def descargar_y_procesar_ciudad(ciudad_nombre, poligono):
    """
    Descargar red de calles y calcular orientaciones para una ciudad
    """
    if poligono is None or poligono.is_empty:
        print(f"    ⚠️  Polígono vacío para {ciudad_nombre}")
        return None, None
    
    try:
        print(f"    📡 Descargando calles para {ciudad_nombre}...")
        
        # Intentar descargar la red de calles con timeout
        import time
        start_time = time.time()
        
        try:
            G = ox.graph_from_polygon(poligono, network_type='drive', simplify=True)
        except Exception as e:
            print(f"    ❌ Error en graph_from_polygon para {ciudad_nombre}: {str(e)[:100]}")
            return None, None
        
        download_time = time.time() - start_time
        print(f"    ⏱️  Descarga completada en {download_time:.1f} segundos")
        
        if G is None or len(G) == 0:
            print(f"    ⚠️  Sin calles para {ciudad_nombre}")
            return None, None
        
        # Convertir a GeoDataFrame
        try:
            nodes, edges = ox.graph_to_gdfs(G)
        except Exception as e:
            print(f"    ❌ Error convirtiendo grafo para {ciudad_nombre}: {str(e)[:100]}")
            return None, None
        
        if edges.empty:
            print(f"    ⚠️  Sin segmentos de calles para {ciudad_nombre}")
            return None, None
        
        print(f"    📏 Procesando {len(edges)} segmentos...")
        
        # Calcular orientación de cada segmento
        def calcular_orientacion(geom):
            if geom is None or geom.is_empty:
                return 0
            coords = list(geom.coords)
            if len(coords) < 2:
                return 0
            
            # Calcular bearing entre primer y último punto
            y1, x1 = coords[0][1], coords[0][0]
            y2, x2 = coords[-1][1], coords[-1][0]
            
            # Fórmula para calcular bearing
            from math import atan2, cos, sin, radians, degrees
            
            # Convertir a radianes
            lat1, lon1 = radians(y1), radians(x1)
            lat2, lon2 = radians(y2), radians(x2)
            
            dlon = lon2 - lon1
            
            # Fórmula de bearing
            x = sin(dlon) * cos(lat2)
            y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            
            bearing = atan2(x, y)
            bearing = degrees(bearing)
            
            # Normalizar a 0-360
            bearing = (bearing + 360) % 360
            
            return bearing
        
        # Aplicar cálculo de orientación en lotes para mejor performance
        chunk_size = 1000
        orientations = []
        for i in range(0, len(edges), chunk_size):
            chunk = edges.iloc[i:i+chunk_size]
            chunk_orientations = chunk['geometry'].apply(calcular_orientacion)
            orientations.extend(chunk_orientations)
        
        edges['orientation'] = orientations
        
        # Agregar metadatos
        edges['ciudad'] = ciudad_nombre
        edges['ciudad_id'] = generar_id_ciudad(ciudad_nombre)
        
        print(f"    ✅ {len(edges)} segmentos procesados para {ciudad_nombre}")
        return edges, G
        
    except Exception as e:
        print(f"    ❌ Error procesando {ciudad_nombre}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None, None

def calcular_orientacion_y_grafico(edges, ciudad_nombre, ciudad_id, poblacion=None):
    """
    Calcular histograma de orientaciones y generar gráfico polar
    """
    if edges is None or len(edges) == 0:
        print(f"    ⚠️  Sin datos para gráfico de {ciudad_nombre}")
        return None, 0
    
    try:
        # Calcular histograma de orientaciones
        orientations = edges['orientation'].dropna()
        
        if len(orientations) == 0:
            print(f"    ⚠️  Sin orientaciones válidas para {ciudad_nombre}")
            return None, 0
        
        print(f"    📊 Generando gráfico para {ciudad_nombre}...")
        
        bin_counts, bin_edges = np.histogram(
            orientations, 
            bins=36,  # 36 bins = 10 grados cada uno
            range=(0, 360)
        )
        
        # Convertir a proporciones
        if bin_counts.sum() > 0:
            bin_proportions = bin_counts / bin_counts.sum()
        else:
            bin_proportions = np.zeros(36)
        
        # Crear gráfico polar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
        
        # Ángulos para los bins (centro de cada bin)
        bin_centers = np.deg2rad(bin_edges[:-1] + 5)  # 5 = 10/2
        
        # Barras en gráfico polar
        bars = ax.bar(bin_centers, bin_proportions, 
                      width=np.deg2rad(10), 
                      alpha=0.7,
                      color='#2E86AB',  # Azul peruano
                      edgecolor='black',
                      linewidth=0.5)
        
        # Configurar gráfico
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        
        # Título con población si está disponible
        if poblacion and poblacion > 0:
            title = f'{ciudad_nombre}\n(Población: {poblacion:,})'
        else:
            title = ciudad_nombre
        
        ax.set_title(title, fontsize=14, pad=20, weight='bold')
        
        # Añadir estadísticas
        stats_text = f"Segmentos: {len(edges):,}"
        
        # Calcular orientación dominante
        if bin_counts.sum() > 0:
            idx_max = np.argmax(bin_counts)
            orient_dom = idx_max * 10  # Grados
            propor_dom = bin_proportions[idx_max]
            
            # Traducir dirección
            direcciones = {
                (0, 22.5): 'Norte', (22.5, 67.5): 'Noreste',
                (67.5, 112.5): 'Este', (112.5, 157.5): 'Sureste',
                (157.5, 202.5): 'Sur', (202.5, 247.5): 'Suroeste',
                (247.5, 292.5): 'Oeste', (292.5, 337.5): 'Noroeste',
                (337.5, 360): 'Norte'
            }
            
            direccion = 'Norte'
            for rango, dir_nombre in direcciones.items():
                if rango[0] <= orient_dom < rango[1]:
                    direccion = dir_nombre
                    break
            
            stats_text += f"\nDominante: {direccion} ({orient_dom}°)"
            stats_text += f"\nProporción: {propor_dom:.1%}"
        
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#F7DC6F', alpha=0.8))
        
        # Añadir rótulos cardinales
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'O'], fontsize=11, weight='bold')
        
        # Ocultar etiquetas radiales para más claridad
        ax.set_yticklabels([])
        
        # Añadir grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Guardar gráfico principal
        grafico_path = WEB_GRAFICOS_DIR / f"{ciudad_id}.png"
        plt.savefig(grafico_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"    💾 Gráfico guardado: {grafico_path.name}")
        
        # Guardar miniatura
        miniatura_path = WEB_GRAFICOS_DIR / f"{ciudad_id}_thumb.png"
        fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(projection='polar'))
        ax.bar(bin_centers, bin_proportions, width=np.deg2rad(10), 
               alpha=0.7, color='#2E86AB')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        
        # Título abreviado para miniatura
        titulo_miniatura = ciudad_nombre[:12] + '...' if len(ciudad_nombre) > 12 else ciudad_nombre
        ax.set_title(titulo_miniatura, fontsize=9, pad=10)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'O'], fontsize=8)
        ax.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(miniatura_path, dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"    💾 Miniatura guardada: {miniatura_path.name}")
        
        return bin_proportions, len(edges)
        
    except Exception as e:
        print(f"    ❌ Error generando gráfico para {ciudad_nombre}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None, 0

def procesar_todas_las_ciudades(gdf, max_ciudades=None):
    """
    Procesar todas las ciudades: descargar calles, calcular orientaciones, generar gráficos
    """
    print("\n" + "="*60)
    print("PROCESANDO CIUDADES Y GENERANDO GRÁFICOS")
    print("="*60)
    
    # Limitar para pruebas
    if max_ciudades:
        print(f"\n⚠️  MODO PRUEBA: procesando {max_ciudades} ciudades")
        gdf = gdf.head(max_ciudades)
    else:
        print(f"\n🚀 MODO COMPLETO: procesando {len(gdf)} ciudades")
    
    ciudades_info = []
    todas_las_calles = []
    exitos = 0
    fallos = 0
    
    # Configurar iteración
    if HAS_TQDM:
        iterador = tqdm(gdf.iterrows(), total=len(gdf), desc="Procesando ciudades")
    else:
        iterador = enumerate(gdf.iterrows())
    
    for idx, row in iterador:
        if not HAS_TQDM:
            print(f"\n[{idx+1}/{len(gdf)}] ", end="")
        
        ciudad_nombre = str(row['ciudad_nombre']) if pd.notna(row['ciudad_nombre']) else f"Ciudad_{idx}"
        ciudad_id = generar_id_ciudad(ciudad_nombre)
        
        if not HAS_TQDM:
            print(f"{ciudad_nombre}")
        
        # Obtener población si está disponible
        poblacion = None
        if 'poblacion' in gdf.columns and pd.notna(row.get('poblacion')):
            poblacion = int(row['poblacion'])
        elif 'pob17' in gdf.columns and pd.notna(row.get('pob17')):
            poblacion = int(row['pob17'])
        
        # Descargar y procesar ciudad
        edges, G = descargar_y_procesar_ciudad(ciudad_nombre, row['geometry'])
        
        if edges is not None and len(edges) > 0:
            # Generar gráfico de orientación
            orientaciones, num_segmentos = calcular_orientacion_y_grafico(
                edges, ciudad_nombre, ciudad_id, poblacion
            )
            
            # Guardar calles
            edges['ciudad_id'] = ciudad_id
            todas_las_calles.append(edges)
            
            # Obtener región natural
            region = ''
            if 'region_natural' in gdf.columns and pd.notna(row.get('region_natural')):
                region = str(row['region_natural']).capitalize()
            elif 'regnat' in gdf.columns and pd.notna(row.get('regnat')):
                region = str(row['regnat']).capitalize()
            
            # Obtener departamento
            departamento = ''
            if 'departamento' in gdf.columns and pd.notna(row.get('departamento')):
                departamento = str(row['departamento'])
            
            # Guardar información de la ciudad
            centroide = row['geometry'].centroid
            ciudad_info = {
                "id": ciudad_id,
                "nombre": ciudad_nombre,
                "departamento": departamento,
                "provincia": row.get('provincia', ''),
                "region_natural": region,
                "poblacion": poblacion or 0,
                "segmentos_calles": int(num_segmentos),
                "ubicacion": {
                    "lat": round(float(centroide.y), 6),
                    "lng": round(float(centroide.x), 6)
                },
                "tiene_datos": True,
                "archivo_grafico": f"{ciudad_id}.png",
                "archivo_miniatura": f"{ciudad_id}_thumb.png"
            }
            
            if orientaciones is not None:
                # Guardar orientación dominante
                idx_max = np.argmax(orientaciones)
                ciudad_info["orientacion_dominante"] = int(idx_max * 10)
                ciudad_info["proporcion_dominante"] = float(orientaciones[idx_max])
            
            ciudades_info.append(ciudad_info)
            exitos += 1
            
        else:
            if not HAS_TQDM:
                print(f"    ⚠️  Sin datos para {ciudad_nombre}")
            
            # Aún así guardar información básica
            centroide = row['geometry'].centroid
            ciudad_info = {
                "id": ciudad_id,
                "nombre": ciudad_nombre,
                "departamento": row.get('departamento', ''),
                "provincia": row.get('provincia', ''),
                "region_natural": row.get('region_natural', row.get('regnat', '')),
                "poblacion": poblacion or 0,
                "segmentos_calles": 0,
                "ubicacion": {
                    "lat": round(float(centroide.y), 6),
                    "lng": round(float(centroide.x), 6)
                },
                "tiene_datos": False,
                "archivo_grafico": "",
                "archivo_miniatura": ""
            }
            ciudades_info.append(ciudad_info)
            fallos += 1
        
        # Pequeña pausa para no sobrecargar la API de OSM
        import time
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print("RESUMEN DEL PROCESAMIENTO:")
    print(f"  ✅ Éxitos: {exitos} ciudades con datos")
    print(f"  ❌ Fallos: {fallos} ciudades sin datos")
    print(f"{'='*60}")
    
    # Combinar todas las calles
    if todas_las_calles:
        todas_las_calles_gdf = gpd.pd.concat(todas_las_calles, ignore_index=True)
        print(f"\n📊 Total segmentos de calles descargados: {len(todas_las_calles_gdf):,}")
    else:
        todas_las_calles_gdf = None
        print("\n⚠️  No se descargaron segmentos de calles")
    
    return ciudades_info, todas_las_calles_gdf

def crear_gpkg_completo(gdf, todas_las_calles_gdf, ciudades_info):
    """Crear GPKG único con múltiples layers"""
    print("\n" + "="*60)
    print("CREANDO ARCHIVO GPKG")
    print("="*60)
    
    gpkg_path = WEB_DATA_DIR / "peru_sno_datos.gpkg"
    
    # Eliminar archivo existente
    if gpkg_path.exists():
        print(f"  🔄 Reemplazando archivo existente: {gpkg_path}")
        gpkg_path.unlink()
    
    # Layer 1: Polígonos urbanos completos
    print("  📁 Layer: poligonos_urbanos")
    gdf.to_file(gpkg_path, layer='poligonos_urbanos', driver='GPKG')
    
    # Layer 2: Calles por ciudad (si las tenemos)
    if todas_las_calles_gdf is not None and len(todas_las_calles_gdf) > 0:
        print("  📁 Layer: calles_todas_ciudades")
        todas_las_calles_gdf.to_file(gpkg_path, layer='calles_todas_ciudades', driver='GPKG')
        
        # También guardar por región natural si tenemos esa info
        region_col = None
        if 'region_natural' in gdf.columns:
            region_col = 'region_natural'
        elif 'regnat' in gdf.columns:
            region_col = 'regnat'
        
        if region_col:
            # Crear diccionario de ciudad -> región
            ciudad_region = {}
            for _, row in gdf.iterrows():
                ciudad_nombre = row['ciudad_nombre']
                region = row[region_col] if pd.notna(row[region_col]) else ''
                ciudad_region[ciudad_nombre] = str(region).capitalize()
            
            # Agregar región a las calles
            todas_las_calles_gdf['region'] = todas_las_calles_gdf['ciudad'].map(ciudad_region)
            
            # Guardar por región
            for region in todas_las_calles_gdf['region'].unique():
                if region and str(region) != 'nan' and region != '':
                    region_norm = str(region).lower().replace(' ', '_')
                    calles_region = todas_las_calles_gdf[todas_las_calles_gdf['region'] == region]
                    
                    if len(calles_region) > 0:
                        layer_name = f"calles_{region_norm}"
                        print(f"  📁 Layer: {layer_name} ({len(calles_region)} segmentos)")
                        calles_region.to_file(gpkg_path, layer=layer_name, driver='GPKG')
    
    if gpkg_path.exists():
        size_mb = gpkg_path.stat().st_size / (1024*1024)
        print(f"\n✅ GPKG creado: {gpkg_path}")
        print(f"   Tamaño: {size_mb:.1f} MB")
    else:
        print("\n⚠️  No se pudo crear el archivo GPKG")

def generar_metadata_json(ciudades_info, gdf):
    """Generar archivo JSON con metadatos para la web"""
    print("\n" + "="*60)
    print("GENERANDO METADATOS JSON")
    print("="*60)
    
    # Ordenar por población descendente
    ciudades_info_sorted = sorted(
        ciudades_info, 
        key=lambda x: x.get('poblacion', 0), 
        reverse=True
    )
    
    # Obtener regiones únicas
    regiones = []
    for ciudad in ciudades_info:
        region = ciudad.get('region_natural', '')
        if region and region not in regiones:
            regiones.append(region)
    
    # Obtener departamentos únicos
    departamentos = []
    for ciudad in ciudades_info:
        depto = ciudad.get('departamento', '')
        if depto and depto not in departamentos:
            departamentos.append(depto)
    
    # Calcular estadísticas
    ciudades_con_datos = sum(1 for c in ciudades_info if c.get('tiene_datos', False))
    total_segmentos = sum(c.get('segmentos_calles', 0) for c in ciudades_info)
    total_poblacion = sum(c.get('poblacion', 0) for c in ciudades_info)
    
    metadata = {
        "ciudades": ciudades_info_sorted,
        "estadisticas": {
            "total_ciudades": len(ciudades_info),
            "ciudades_con_datos": ciudades_con_datos,
            "ciudades_sin_datos": len(ciudades_info) - ciudades_con_datos,
            "total_segmentos": total_segmentos,
            "poblacion_total": total_poblacion,
            "poblacion_promedio": int(np.mean([c.get('poblacion', 0) for c in ciudades_info]) if ciudades_info else 0)
        },
        "regiones_naturales": sorted(regiones),
        "departamentos": sorted(departamentos),
        "actualizado": pd.Timestamp.now().isoformat(),
        "fuente": "OpenStreetMap via OSMNx",
        "procesado_por": "dataprep_web_data.py"
    }
    
    # Guardar JSON
    json_path = WEB_DATA_DIR / "ciudades.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Metadata generada: {json_path}")
    print(f"   • {len(ciudades_info)} ciudades")
    print(f"   • {ciudades_con_datos} con datos de calles")
    print(f"   • {total_segmentos:,} segmentos totales")
    
    return metadata

def generar_estructura_web():
    """Generar estructura básica de archivos web"""
    print("\nGenerando estructura de archivos web...")
    
    # Archivos necesarios para GitHub Pages
    archivos_base = [
        ".nojekyll",
        "CNAME",
        "README.md"
    ]
    
    for archivo in archivos_base:
        if not (WEB_DIR / archivo).exists():
            if archivo == ".nojekyll":
                with open(WEB_DIR / archivo, "w") as f:
                    f.write("# Evita que GitHub Pages use Jekyll\n")
            elif archivo == "CNAME":
                with open(WEB_DIR / archivo, "w") as f:
                    f.write("# Agrega tu dominio personalizado aquí\n")
            elif archivo == "README.md":
                with open(WEB_DIR / archivo, "w", encoding="utf-8") as f:
                    f.write("""# Aplicación Web - Orientación de Calles del Perú

Esta carpeta contiene la aplicación web estática para visualizar y descargar los datos de orientación de calles de las principales ciudades del Perú.

Los datos han sido generados automáticamente usando OSMNx.
""")
    
    print("Estructura web generada")

def main():
    """Función principal"""
    print("=" * 70)
    print("GENERADOR DE DATOS WEB - ORIENTACIÓN DE CALLES DEL PERÚ")
    print("=" * 70)
    print("\nEste script:")
    print("1. Descarga redes de calles de OpenStreetMap")
    print("2. Calcula orientaciones y genera gráficos polares")
    print("3. Crea metadatos JSON para filtrado web")
    print("4. Genera archivo GPKG con todos los datos")
    print("=" * 70)
    
    # Preguntar modo (prueba/completo)
    print("\n⚠️  ADVERTENCIA: Descargar datos de 91 ciudades puede tardar HORAS")
    print("               y consumir mucho ancho de banda.\n")
    
    print("Modos disponibles:")
    print("1. Modo completo (todas las ciudades) - Tarda varias horas")
    print("2. Modo prueba (primeras 5 ciudades) - Recomendado para desarrollo")
    print("3. Modo ultra-prueba (solo 1 ciudad) - Para verificar funcionamiento")
    
    while True:
        try:
            modo = input("\nSelecciona modo [1/2/3] (Enter para modo prueba): ").strip()
            if modo == "" or modo == "2":
                MAX_CIUDADES = 5
                print("✓ Modo prueba (5 ciudades)")
                break
            elif modo == "1":
                MAX_CIUDADES = None
                print("✓ Modo completo (91 ciudades)")
                confirmar = input("  ⚠️  ¿Estás seguro? Esto puede tardar horas [s/N]: ")
                if confirmar.lower() != 's':
                    print("  Volviendo al modo prueba...")
                    MAX_CIUDADES = 5
                break
            elif modo == "3":
                MAX_CIUDADES = 1
                print("✓ Modo ultra-prueba (1 ciudad)")
                break
            else:
                print("Opción no válida")
        except KeyboardInterrupt:
            print("\nProceso cancelado")
            return
    
    print("\n" + "-" * 50)
    
    try:
        # 1. Procesar polígonos urbanos
        gdf = procesar_poligonos_urbanos()
        
        # 2. Procesar ciudades seleccionadas
        ciudades_info, todas_las_calles_gdf = procesar_todas_las_ciudades(gdf, MAX_CIUDADES)
        
        # 3. Generar metadata JSON
        metadata = generar_metadata_json(ciudades_info, gdf)
        
        # 4. Crear GPKG único
        crear_gpkg_completo(gdf, todas_las_calles_gdf, ciudades_info)
        
        # 5. Generar estructura web
        generar_estructura_web()
        
        # Reporte final
        print("\n" + "=" * 70)
        print("✅ ¡PROCESO COMPLETADO!")
        print("=" * 70)
        
        print(f"\n📊 RESULTADOS:")
        print(f"  • {len(ciudades_info)} ciudades procesadas")
        print(f"  • {metadata['estadisticas']['ciudades_con_datos']} con datos de calles")
        print(f"  • {metadata['estadisticas']['total_segmentos']:,} segmentos de calles")
        
        # Contar gráficos generados
        graficos = list(WEB_GRAFICOS_DIR.glob("*.png"))
        thumbnails = list(WEB_GRAFICOS_DIR.glob("*_thumb.png"))
        print(f"  • {len(graficos)} gráficos generados")
        print(f"  • {len(thumbnails)} miniaturas generadas")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        for archivo in WEB_DATA_DIR.glob("*"):
            if archivo.is_file():
                if archivo.suffix != '.json':  # No mostrar tamaño de JSON
                    size_mb = archivo.stat().st_size / (1024*1024)
                    print(f"  • {archivo.name} ({size_mb:.1f} MB)")
                else:
                    print(f"  • {archivo.name}")
        
        print(f"\n🎨 GRÁFICOS en {WEB_GRAFICOS_DIR}:")
        graficos_main = [g for g in graficos if '_thumb' not in g.name]
        for i, g in enumerate(graficos_main[:3]):  # Mostrar primeros 3
            print(f"  • {g.name}")
        if len(graficos_main) > 3:
            print(f"  • ... y {len(graficos_main)-3} más")
        
        print("\n🚀 PRÓXIMOS PASOS:")
        print("1. Configurar GitHub Pages: Settings > Pages > Source: /web")
        print("2. La web estará en: https://[usuario].github.io/peru_street_network_orientation/")
        print("3. Para regenerar datos: .venv\Scripts\python.exe dataprep_web_data.py")
        
        # Mostrar ejemplo de ciudad con datos
        for ciudad in ciudades_info:
            if ciudad.get('tiene_datos', False):
                print(f"\n📍 EJEMPLO - {ciudad['nombre']}:")
                print(f"  Población: {ciudad['poblacion']:,}")
                print(f"  Segmentos: {ciudad['segmentos_calles']:,}")
                print(f"  Región: {ciudad['region_natural']}")
                print(f"  Gráfico: {ciudad['archivo_grafico']}")
                break
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()