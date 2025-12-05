// app.js - Lógica principal de la web

// Rutas de datos
const DATA_PATH = 'data/ciudades.json';
const GPKG_PATH = 'data/peru_sno_datos.gpkg';
const GRAFICOS_PATH = 'data/graficos/';

// Estado global
let ciudades = [];
let regiones = [];
let departamentos = [];
let provincias = [];
let destacados = [];
let seleccionadas = [];

// Utilidades
function formatearNumero(n) {
  return n ? n.toLocaleString('es-PE') : '-';
}

function cargarIntro() {
  fetch('README.md')
    .then(r => r.text())
    .then(md => {
      // Mostrar solo el primer bloque (introducción)
      const intro = md.split('\n##')[0];
      document.getElementById('intro').innerHTML = marked.parse(intro);
    });
}

function cargarDatos() {
  fetch(DATA_PATH)
    .then(r => r.json())
    .then(data => {
      ciudades = data.ciudades;
      regiones = data.regiones_naturales;
      departamentos = data.departamentos;
      destacados = seleccionarDestacados(ciudades);
      renderDestacados();
      renderGaleriaCompleta();
      renderFiltros();
      renderDescargaFiltros();
    });
}

function seleccionarDestacados(ciudades) {
  // Selección simple: 5 ciudades con más segmentos y 5 con menos
  const ordenadas = [...ciudades].sort((a,b) => b.segmentos_calles - a.segmentos_calles);
  return [ordenadas[0], ordenadas[1], ordenadas[2], ordenadas[ordenadas.length-1], ordenadas[ordenadas.length-2]];
}

function renderDestacados() {
  const cont = document.getElementById('destacados');
  cont.innerHTML = '';
  destacados.forEach(c => cont.appendChild(crearMiniatura(c)));
}

function renderGaleriaCompleta() {
  const cont = document.getElementById('galeria-completa');
  cont.innerHTML = '';
  ciudades.forEach(c => cont.appendChild(crearMiniatura(c)));
}

function crearMiniatura(ciudad) {
  const div = document.createElement('div');
  div.className = 'ciudad';
  div.onclick = () => mostrarModalCiudad(ciudad);
  const img = document.createElement('img');
  img.src = GRAFICOS_PATH + ciudad.archivo_miniatura;
  img.alt = ciudad.nombre;
  const nombre = document.createElement('div');
  nombre.className = 'nombre';
  nombre.textContent = ciudad.nombre;
  const region = document.createElement('div');
  region.className = 'region';
  region.textContent = ciudad.region_natural;
  const poblacion = document.createElement('div');
  poblacion.className = 'poblacion';
  poblacion.textContent = 'Población: ' + formatearNumero(ciudad.poblacion);
  div.appendChild(img);
  div.appendChild(nombre);
  div.appendChild(region);
  div.appendChild(poblacion);
  return div;
}

function mostrarModalCiudad(ciudad) {
  const modal = document.getElementById('modal');
  const body = document.getElementById('modal-body');
  body.innerHTML = `
    <img src="${GRAFICOS_PATH + ciudad.archivo_grafico}" alt="${ciudad.nombre}">
    <div class="metadatos">
      <strong>${ciudad.nombre}</strong><br>
      Región: ${ciudad.region_natural}<br>
      Departamento: ${ciudad.departamento}<br>
      Provincia: ${ciudad.provincia}<br>
      Población: ${formatearNumero(ciudad.poblacion)}<br>
      Segmentos de calles: ${formatearNumero(ciudad.segmentos_calles)}
    </div>
    <a class="descargar" href="${GRAFICOS_PATH + ciudad.archivo_grafico}" download>Descargar gráfico PNG</a>
  `;
  modal.classList.remove('hidden');
}

document.getElementById('modal-close').onclick = () => {
  document.getElementById('modal').classList.add('hidden');
};

// Filtros combinados
function renderFiltros() {
  const cont = document.getElementById('filtros');
  cont.innerHTML = '';
  // Región natural
  cont.appendChild(crearFiltroCheckbox('Región natural', regiones, 'region_natural'));
  // Departamento
  cont.appendChild(crearFiltroCheckbox('Departamento', departamentos, 'departamento'));
  // Provincia
  const provinciasUnicas = [...new Set(ciudades.map(c => c.provincia).filter(Boolean))].sort();
  cont.appendChild(crearFiltroCheckbox('Provincia', provinciasUnicas, 'provincia'));
  // Ciudad
  cont.appendChild(crearFiltroCheckbox('Ciudad', ciudades.map(c => c.nombre), 'ciudad'));
  // Botón aplicar
  const btn = document.createElement('button');
  btn.textContent = 'Aplicar filtros';
  btn.onclick = aplicarFiltros;
  cont.appendChild(btn);
}

function crearFiltroCheckbox(label, opciones, campo) {
  const group = document.createElement('div');
  group.className = 'filtro-group';
  const lbl = document.createElement('label');
  lbl.textContent = label + ':';
  group.appendChild(lbl);
  const box = document.createElement('div');
  box.className = 'checkboxes';
  opciones.forEach(op => {
    const id = campo + '_' + op.replace(/\s/g,'_');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.id = id;
    input.value = op;
    input.dataset.campo = campo;
    box.appendChild(input);
    const span = document.createElement('span');
    span.textContent = op;
    box.appendChild(span);
  });
  group.appendChild(box);
  return group;
}

function aplicarFiltros() {
  // Leer checkboxes
  const checks = document.querySelectorAll('#filtros input[type="checkbox"]:checked');
  let filtros = { region_natural: [], departamento: [], provincia: [], ciudad: [] };
  checks.forEach(chk => {
    const campo = chk.dataset.campo;
    filtros[campo].push(chk.value);
  });
  // Filtrar ciudades
  let filtradas = ciudades.filter(c => {
    return (
      (filtros.region_natural.length === 0 || filtros.region_natural.includes(c.region_natural)) &&
      (filtros.departamento.length === 0 || filtros.departamento.includes(c.departamento)) &&
      (filtros.provincia.length === 0 || filtros.provincia.includes(c.provincia)) &&
      (filtros.ciudad.length === 0 || filtros.ciudad.includes(c.nombre))
    );
  });
  seleccionadas = filtradas;
  renderGaleriaFiltrada();
}

function renderGaleriaFiltrada() {
  const cont = document.getElementById('galeria-filtrada');
  cont.innerHTML = '';
  seleccionadas.forEach(c => cont.appendChild(crearMiniatura(c)));
}

// Descargas
function renderDescargaFiltros() {
  const cont = document.getElementById('descarga-filtros');
  cont.innerHTML = '';
  // Reutilizar filtros
  cont.appendChild(crearFiltroCheckbox('Región natural', regiones, 'region_natural'));
  cont.appendChild(crearFiltroCheckbox('Departamento', departamentos, 'departamento'));
  const provinciasUnicas = [...new Set(ciudades.map(c => c.provincia).filter(Boolean))].sort();
  cont.appendChild(crearFiltroCheckbox('Provincia', provinciasUnicas, 'provincia'));
  cont.appendChild(crearFiltroCheckbox('Ciudad', ciudades.map(c => c.nombre), 'ciudad'));
  const btn = document.createElement('button');
  btn.textContent = 'Seleccionar ciudades';
  btn.onclick = aplicarDescargaFiltros;
  cont.appendChild(btn);
}

function aplicarDescargaFiltros() {
  // Igual que aplicarFiltros pero para descargas
  const checks = document.querySelectorAll('#descarga-filtros input[type="checkbox"]:checked');
  let filtros = { region_natural: [], departamento: [], provincia: [], ciudad: [] };
  checks.forEach(chk => {
    const campo = chk.dataset.campo;
    filtros[campo].push(chk.value);
  });
  let filtradas = ciudades.filter(c => {
    return (
      (filtros.region_natural.length === 0 || filtros.region_natural.includes(c.region_natural)) &&
      (filtros.departamento.length === 0 || filtros.departamento.includes(c.departamento)) &&
      (filtros.provincia.length === 0 || filtros.provincia.includes(c.provincia)) &&
      (filtros.ciudad.length === 0 || filtros.ciudad.includes(c.nombre))
    );
  });
  seleccionadas = filtradas;
  renderDescargaPreview();
  renderDescargaOpciones();
}

function renderDescargaPreview() {
  const cont = document.getElementById('descarga-preview');
  cont.innerHTML = '<h3>Ciudades seleccionadas:</h3>';
  const gal = document.createElement('div');
  gal.className = 'galeria';
  seleccionadas.forEach(c => gal.appendChild(crearMiniatura(c)));
  cont.appendChild(gal);
}

function renderDescargaOpciones() {
  const cont = document.getElementById('descarga-opciones');
  cont.innerHTML = '';
  // Botones de descarga
  const btnGPKG = document.createElement('button');
  btnGPKG.textContent = 'Descargar polígonos urbanos (GPKG)';
  btnGPKG.onclick = () => window.open(GPKG_PATH, '_blank');
  cont.appendChild(btnGPKG);
  const btnCalles = document.createElement('button');
  btnCalles.textContent = 'Descargar red de calles (GPKG)';
  btnCalles.onclick = () => window.open(GPKG_PATH, '_blank');
  cont.appendChild(btnCalles);
  const btnGraficos = document.createElement('button');
  btnGraficos.textContent = 'Descargar gráficos (ZIP)';
  btnGraficos.onclick = descargarGraficosZIP;
  cont.appendChild(btnGraficos);
}

function descargarGraficosZIP() {
  // Solo posible si se usa JSZip (no incluido por defecto)
  alert('Funcionalidad ZIP: requiere JSZip. Descarga manual de gráficos disponible.');
}

// Navegación
function mostrarSeccion(id) {
  ['inicio','galeria','descargas'].forEach(sec => {
    document.getElementById(sec).classList.add('hidden');
  });
  document.getElementById(id).classList.remove('hidden');
}
document.getElementById('btnInicio').onclick = () => mostrarSeccion('inicio');
document.getElementById('btnGaleria').onclick = () => mostrarSeccion('galeria');
document.getElementById('btnDescargas').onclick = () => mostrarSeccion('descargas');

// Inicialización
window.onload = () => {
  cargarIntro();
  cargarDatos();
};
