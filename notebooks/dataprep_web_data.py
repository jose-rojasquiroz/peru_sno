import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

INPUT_GPKG = 'data/pu-principalesciudades-WGS84.gpkg'
OUTPUT_DIR = Path('docs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Crear subdirectorios
(OUTPUT_DIR / 'fichas').mkdir(exist_ok=True)
(OUTPUT_DIR / 'data').mkdir(exist_ok=True)
(OUTPUT_DIR / 'graficos').mkdir(exist_ok=True)
(OUTPUT_DIR / 'mapas').mkdir(exist_ok=True)

# Colores por región
COLORES = {
    'Costa': '#c7cb52',
    'Sierra': '#98692e',
    'Selva': '#62a162'
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calcular_densidad(gdf):
    """Calcula el área en km² y la densidad poblacional"""
    # Convertir a CRS métrico (UTM zone 18S para Perú)
    gdf_metric = gdf.to_crs('EPSG:32718')
    gdf['area_km2'] = gdf_metric.geometry.area / 1_000_000
    gdf['densidad'] = gdf['POB17'] / gdf['area_km2']
    return gdf

def get_bearings_from_polygon(city_polygon, city_name):
    """Obtiene los bearings de las calles de una ciudad"""
    try:
        print(f"  Descargando red de calles para {city_name}...")
        graph = ox.graph_from_polygon(city_polygon, network_type='drive')
        graph = ox.bearing.add_edge_bearings(graph)
        bearings = [d['bearing'] for u, v, k, d in graph.edges(keys=True, data=True) if 'bearing' in d]
        print(f"  ✓ {city_name}: {len(bearings)} segmentos de calles")
        return bearings, graph
    except Exception as e:
        print(f"  ✗ Error en {city_name}: {e}")
        return None, None

def create_street_map(graph, city_name, region, output_path):
    """Crea y guarda el mapa de la red de calles"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        
        # Plot del grafo
        ox.plot_graph(graph, ax=ax, node_size=0, edge_color='#333333', 
                      edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        
        ax.set_title(f'{city_name}\nRed de Calles', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Remover ejes
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Mapa guardado: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error creando mapa para {city_name}: {e}")
        return False

def create_polar_plot(bearings, city_name, region, output_path):
    """Crea y guarda el gráfico polar de orientación"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    color = COLORES[region]
    direcciones = ['N', '', 'E', '', 'S', '', 'O', '']
    
    # Histograma
    ax.hist(np.deg2rad(bearings), bins=36, color=color, alpha=0.75, edgecolor='white', linewidth=1)
    
    # Configuración del gráfico
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))
    ax.set_xticklabels(direcciones, fontsize=14, fontweight='bold')
    ax.set_title(f'{city_name}\nOrientación de Calles', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Fondo transparente
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Gráfico guardado: {output_path.name}")

def create_ficha(city_data, bearings, graph, output_path):
    """Crea una ficha completa (imagen) de la ciudad con mapa de calles"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    color = COLORES[city_data['REGNAT']]
    
    # Título principal
    fig.suptitle(f"{city_data['CIUDAD']}", fontsize=28, fontweight='bold', y=0.98)
    
    # MAPA DE CALLES (izquierda, ocupa 2 filas)
    ax_mapa = fig.add_subplot(gs[:, 0])
    try:
        ox.plot_graph(graph, ax=ax_mapa, node_size=0, edge_color='#333333', 
                      edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        ax_mapa.set_title('Red de Calles', fontsize=14, fontweight='bold', pad=10)
        ax_mapa.set_axis_off()
    except:
        ax_mapa.text(0.5, 0.5, 'Mapa no disponible', ha='center', va='center')
        ax_mapa.set_axis_off()
    
    # GRÁFICO POLAR (centro arriba)
    ax_polar = fig.add_subplot(gs[0, 1], projection='polar')
    direcciones = ['N', '', 'E', '', 'S', '', 'O', '']
    ax_polar.hist(np.deg2rad(bearings), bins=36, color=color, alpha=0.75, 
                  edgecolor='white', linewidth=1)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_yticklabels([])
    ax_polar.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))
    ax_polar.set_xticklabels(direcciones, fontsize=11, fontweight='bold')
    ax_polar.set_title('Orientación de Calles', fontsize=13, fontweight='bold', pad=15)
    
    # INFORMACIÓN BÁSICA (derecha arriba)
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis('off')
    
    info_text = f"""
REGIÓN
{city_data['REGNAT']}

DEPARTAMENTO
{city_data['DEPARTAMENTO']}

PROVINCIA
{city_data['PROVINCIA']}

POBLACIÓN (2017)
{city_data['POB17']:,} hab

ÁREA
{city_data['area_km2']:.2f} km²

DENSIDAD
{city_data['densidad']:.0f} hab/km²
"""
    
    ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.15))
    
    # ESTADÍSTICAS DE ORIENTACIÓN (centro abajo)
    ax_stats = fig.add_subplot(gs[1, 1:])
    ax_stats.axis('off')
    
    # Calcular orientaciones principales
    bins = np.linspace(0, 360, 37)
    hist, _ = np.histogram(bearings, bins=bins)
    orientaciones = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    orientaciones_completas = []
    for i in range(8):
        start_idx = i * 4
        end_idx = start_idx + 5
        count = hist[start_idx:end_idx].sum()
        orientaciones_completas.append((orientaciones[i], count))
    
    orientaciones_completas.sort(key=lambda x: x[1], reverse=True)
    
    stats_text = "ORIENTACIONES PRINCIPALES\n\n"
    for i, (orient, count) in enumerate(orientaciones_completas[:4], 1):
        porcentaje = (count / len(bearings)) * 100
        stats_text += f"{i}. {orient:3s}: {porcentaje:5.1f}%  ({count:,} segmentos)\n"
    
    stats_text += f"\nTOTAL DE SEGMENTOS: {len(bearings):,}"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Footer
    fig.text(0.5, 0.01, 'Elaborado por José Rojas-Quiroz | Datos: INEI 2017, OpenStreetMap', 
             ha='center', fontsize=10, color='gray')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Ficha guardada: {output_path.name}")

def process_city(idx, city_row, gdf_cities):
    """Procesa una ciudad individual"""
    city_name = city_row['CIUDAD']
    city_polygon = city_row['geometry']
    region = city_row['REGNAT']
    
    print(f"\n[{idx+1}/{len(gdf_cities)}] Procesando: {city_name} ({region})")
    
    # Obtener bearings y grafo
    bearings, graph = get_bearings_from_polygon(city_polygon, city_name)
    
    if bearings is None or len(bearings) == 0:
        print(f"  ✗ Sin datos para {city_name}")
        return None
    
    # Crear mapa de calles
    mapa_path = OUTPUT_DIR / 'mapas' / f'{city_name.replace(" ", "_")}_mapa.png'
    create_street_map(graph, city_name, region, mapa_path)
    
    # Crear gráfico polar individual
    polar_path = OUTPUT_DIR / 'graficos' / f'{city_name.replace(" ", "_")}_polar.png'
    create_polar_plot(bearings, city_name, region, polar_path)
    
    # Crear ficha completa (ahora incluye el mapa)
    ficha_path = OUTPUT_DIR / 'fichas' / f'{city_name.replace(" ", "_")}_ficha.png'
    create_ficha(city_row, bearings, graph, ficha_path)
    
    # Preparar datos para JSON
    city_data = {
        'nombre': city_name,
        'region': region,
        'departamento': city_row['DEPARTAMENTO'],
        'provincia': city_row['PROVINCIA'],
        'poblacion': int(city_row['POB17']),
        'area_km2': float(city_row['area_km2']),
        'densidad': float(city_row['densidad']),
        'bearings': [float(b) for b in bearings],
        'num_segmentos': len(bearings),
        'mapa_calles': f'mapas/{city_name.replace(" ", "_")}_mapa.png',
        'grafico_polar': f'graficos/{city_name.replace(" ", "_")}_polar.png',
        'ficha': f'fichas/{city_name.replace(" ", "_")}_ficha.png'
    }
    
    return city_data, graph

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("="*70)
    print("PRE-PROCESAMIENTO DE DATOS - CIUDADES DEL PERÚ")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos del GeoPackage...")
    gdf_cities = gpd.read_file(INPUT_GPKG)
    gdf_cities['geometry'] = gdf_cities['geometry'].buffer(0)
    gdf_cities = gdf_cities[gdf_cities.geometry.is_valid]
    print(f"  ✓ {len(gdf_cities)} ciudades cargadas")
    
    # 2. Calcular densidades
    print("\n[2/5] Calculando áreas y densidades poblacionales...")
    gdf_cities = calcular_densidad(gdf_cities)
    print(f"  ✓ Densidades calculadas")
    
    # 3. Procesar ciudades
    print("\n[3/5] Procesando ciudades (descargando redes y generando gráficos/mapas)...")
    print("  Nota: Este proceso puede tomar varios minutos...\n")
    
    all_city_data = []
    all_graphs = []
    
    # Procesamiento secuencial (más estable que paralelo para OSMnx)
    for idx, (_, city_row) in enumerate(gdf_cities.iterrows()):
        result = process_city(idx, city_row, gdf_cities)
        if result:
            city_data, graph = result
            all_city_data.append(city_data)
            all_graphs.append((city_data['nombre'], graph))
    
    print(f"\n  ✓ {len(all_city_data)} ciudades procesadas exitosamente")
    
    # 4. Guardar datos JSON
    print("\n[4/5] Guardando datos JSON...")
    
    # JSON general con todas las ciudades
    with open(OUTPUT_DIR / 'data' / 'ciudades.json', 'w', encoding='utf-8') as f:
        json.dump(all_city_data, f, ensure_ascii=False, indent=2)
    print("  ✓ ciudades.json")
    
    # JSON por región (top 3)
    for region in ['Costa', 'Sierra', 'Selva']:
        region_data = [c for c in all_city_data if c['region'] == region]
        region_data.sort(key=lambda x: x['poblacion'], reverse=True)
        top_3 = region_data[:3]
        
        with open(OUTPUT_DIR / 'data' / f'{region.lower()}_top3.json', 'w', encoding='utf-8') as f:
            json.dump(top_3, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {region.lower()}_top3.json")
    
    # 5. Exportar GeoPackages
    print("\n[5/5] Exportando GeoPackages...")
    
    # GeoPackage de polígonos urbanos
    gdf_export = gdf_cities[['CIUDAD', 'POB17', 'REGNAT', 'DEPARTAMENTO', 
                              'PROVINCIA', 'area_km2', 'densidad', 'geometry']].copy()
    gdf_export.to_file(OUTPUT_DIR / 'data' / 'poligonos_urbanos.gpkg', driver='GPKG')
    print("  ✓ poligonos_urbanos.gpkg")
    
    # GeoPackage de red de calles (todas las ciudades)
    print("  Consolidando redes de calles...")
    all_edges = []
    
    for city_name, graph in all_graphs:
        if graph is not None:
            try:
                edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
                edges['ciudad_nombre'] = city_name
                all_edges.append(edges)
            except Exception as e:
                print(f"  ✗ Error exportando calles de {city_name}: {e}")
    
    if all_edges:
        gdf_streets = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True))
        # Mantener solo columnas relevantes
        cols_to_keep = ['ciudad_nombre', 'geometry', 'bearing', 'length', 
                        'highway', 'name', 'oneway']
        cols_available = [c for c in cols_to_keep if c in gdf_streets.columns]
        gdf_streets = gdf_streets[cols_available]
        
        gdf_streets.to_file(OUTPUT_DIR / 'data' / 'red_calles.gpkg', driver='GPKG')
        print(f"  ✓ red_calles.gpkg ({len(gdf_streets)} segmentos)")
    
    # Resumen final
    print("\n" + "="*70)
    print("✓ PRE-PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nArchivos generados en: {OUTPUT_DIR.absolute()}")
    print(f"\n  Fichas:          {len(list((OUTPUT_DIR / 'fichas').glob('*.png')))} archivos PNG")
    print(f"  Gráficos:        {len(list((OUTPUT_DIR / 'graficos').glob('*.png')))} archivos PNG")
    print(f"  Mapas:           {len(list((OUTPUT_DIR / 'mapas').glob('*.png')))} archivos PNG")
    print(f"  Datos JSON:      {len(list((OUTPUT_DIR / 'data').glob('*.json')))} archivos JSON")
    print(f"  GeoPackages:     {len(list((OUTPUT_DIR / 'data').glob('*.gpkg')))} archivos GPKG")
    print("\n¡Ahora puedes usar estos archivos en tu aplicación web!")

if __name__ == "__main__":
    main()