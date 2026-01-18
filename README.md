# H3 Canibalización - Análisis de Comportamiento de Clientes

## Descripción General

Proyecto de análisis de datos para identificar patrones de canibalización en compras de sorteos. El sistema procesa información histórica de compras y sorteos para clasificar el comportamiento de los clientes y detectar casos donde clientes habituales dejan de comprar o migran a otros productos.

## Estructura del Proyecto

```
.
├── AnalisisV5.py                    # Script principal de análisis completo
├── AnalisisV5FilterClients.py       # Análisis con filtro de clientes recurrentes
├── RollUpFilterClients.py           # Carga de rollup filtrado a BigQuery
├── pyproject.toml                   # Configuración de dependencias
├── .python-version                  # Versión de Python (3.12)
├── .gitignore                       # Archivos excluidos del control de versiones
└── uv.lock                          # Lock file de dependencias
```

## Requisitos

- Python 3.12
- Credenciales de Google Cloud con acceso a BigQuery
- Archivo de credenciales: `sorteostec-ml-5f178b142b6f.json`

### Dependencias principales

- `pandas>=2.3.3`
- `google-cloud-bigquery>=3.38.0`
- `google-cloud-bigquery-storage>=2.34.0`
- `numpy>=2.3.3`
- `pyarrow>=21.0.0`
- `google-auth>=2.41.1`

## Scripts Principales

### 1. AnalisisV5.py

Script completo que ejecuta todo el pipeline de análisis:

- **Entrada**: Tablas de BigQuery
- **Proceso**: Filtrado temporal, enriquecimiento, análisis de ventanas, clasificación
- **Salida**: 
  - CSV: `/home/sam.salinas/PythonProjects/H3Canibalization/rollup_canibalizacion.csv`
  - BigQuery: `sorteostec-ml.h3.rollup_canibalizacion`

### 2. AnalisisV5FilterClients.py

Variante del análisis que filtra únicamente clientes con historial de compras en múltiples ediciones:

- **Diferencia clave**: Filtra registros donde `hist_ediciones_distintas >= 2`
- **Salida**:
  - CSV: `rollup_canibalizacionFilterClients.csv`
  - BigQuery: `sorteostec-ml.h3.rollup_canibalizacion_filter_clients`

### 3. RollUpFilterClients.py

Script simplificado para procesar y cargar un rollup ya generado:

- **Entrada**: CSV `rollup_canibalizacion.csv`
- **Proceso**: Filtrado por `hist_ediciones_distintas >= 2`
- **Salida**: BigQuery `sorteostec-ml.h3.rollup_canibalizacion_filter_clients`

## Flujo de Datos

1. **Extracción**: Descarga de datos desde BigQuery
2. **Filtrado temporal**: Ventana de 2 años para análisis
3. **Enriquecimiento**: Merge con información de sorteos
4. **Análisis de ventanas**: Identificación de períodos pre-cierre y semana de cierre
5. **Agregación**: Rollup por cliente y tipo de producto
6. **Clasificación**: Categorización del comportamiento de canibalización
7. **Carga**: Almacenamiento en BigQuery

## Logs

El sistema genera logs detallados en:
- Directorio: `/home/sam.salinas/PythonProjects/H3Canibalization/logs/`
- Archivo: `h3canibalization.log`
- Configuración: Rotating file handler (10 MB x 5 archivos)

## Ejecución

```bash
# Análisis completo
python AnalisisV5.py

# Análisis con filtro de clientes
python AnalisisV5FilterClients.py

# Solo carga de rollup filtrado
python RollUpFilterClients.py
```

## Archivos Generados

Durante la ejecución se generan los siguientes archivos temporales:

- `detalle_compra.csv` - Datos crudos de compras
- `sorteo.csv` - Datos crudos de sorteos
- `detalle_compra_2_años.csv` - Compras filtradas por ventana temporal
- `sorteo_2_años.csv` - Sorteos filtrados por ventana temporal
- `rollup_canibalizacion.csv` - Resultado final del análisis
- `rollup_canibalizacionFilterClients.csv` - Resultado filtrado

**Nota**: Estos archivos CSV están excluidos del control de versiones según `.gitignore`.