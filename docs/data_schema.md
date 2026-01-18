# Esquema de Datos y Flujo del Sistema

## Diagrama de Flujo de Datos

```
┌─────────────────────────────────┐
│  BigQuery: detalle_compra       │
│  BigQuery: sorteo                │
└────────────┬────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │  Descarga a CSV    │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Filtro 2 años      │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Merge y Enriqueci- │
    │ miento de datos    │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Marcado de ventanas│
    │ (pre-cierre/cierre)│
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Construcción de    │
    │ ventanas anuales   │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Agregación por     │
    │ persona×producto   │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Clasificación de   │
    │ canibalización     │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Filtro opcional        │
    │ (hist_ediciones >= 2)  │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │  CSV: rollup_canibalizacion     │
    │  BigQuery: rollup_canibalizacion│
    └────────────────────────────────┘
```

## Transformaciones de Datos

### Etapa 1: Datos Crudos

**Input**: Tablas originales de BigQuery

```
detalle_compra (original)
├── clave_detalle_compra
├── agrupador_interno_compra
├── clave_persona
├── clave_rotulo
├── clave_edicion_producto
├── fecha_rotulacion_finalizacion
├── clave_boleto_unico
├── clave_boleto_intranet
└── numero_boleto

sorteo (original)
├── clave_edicion_producto
├── tipo_producto
├── numero_sorteo
├── desc_sorteo
├── fecha_inicio
├── fecha_fin
├── fecha_celebracion
└── [otras columnas no utilizadas]
```

### Etapa 2: Datos Filtrados (2 años)

**Transformación**: 
- Conversión de tipos (tipo_producto, numero_sorteo a int)
- Parseo de fechas
- Filtro temporal: `fecha_fin` entre (hoy - 730 días) y hoy

```
dfDetalleCompraFiltrado
├── [columnas de detalle_compra]
└── [sin item_completo, tipo_producto, numero_sorteo - se eliminan después del merge]

dfSorteoFiltrado
├── [columnas de sorteo]
└── item_completo (generado: desc_sorteo + numero_sorteo)
```

### Etapa 3: Datos Enriquecidos

**Transformación**: Merge de detalle con sorteo + marcado de ventanas

```
df_enriq
├── [columnas de detalle_compra]
├── tipo_producto
├── numero_sorteo
├── desc_sorteo
├── fecha_inicio
├── fecha_fin
├── fecha_celebracion
├── semana_cierre_ini (calculado: fecha_fin - 7 días)
├── en_semana_cierre (boolean)
└── en_pre_cierre (boolean)
```

### Etapa 4: Datos Agregados (Rollup)

**Transformación**: Agregación por persona×producto×ventana

```
rollup_df
├── ventana_numero
├── clave_persona
├── tipo_producto
├── numero_sorteo_ultima
├── clave_edicion_producto_ultima
├── ultima_fecha_inicio
├── ultima_fecha_fin
├── ultima_semana_cierre_ini
├── ultima_semana_cierre_fin
├── hist_boletos_pre_cierre
├── hist_boletos_semana_cierre
├── hist_ediciones_distintas
├── ultima_boletos_pre_cierre
├── ultima_boletos_semana_cierre
├── compro_ultima_edicion
├── otros_boletos_en_semana_cierre_del_producto
├── otros_boletos_fuera_semana_cierre_del_producto
├── otros_tipos_producto_distintos
├── canibalizacion_clase
├── hist_detalle_por_edicion_json
└── otros_detalle_en_ventana_ultima_json
```

## Estructuras JSON Embebidas

### hist_detalle_por_edicion_json

Array de objetos con el siguiente esquema:

```json
[
  {
    "numero_sorteo": 123,
    "tipo_producto": "1",
    "id_sorteo": "1-123",
    "clave_edicion_producto": 456,
    "boletos_pre_cierre": 2,
    "boletos_semana_cierre": 1,
    "fecha_fin_edicion": "2024-01-15"
  }
]
```

### otros_detalle_en_ventana_ultima_json

Array de objetos con el siguiente esquema:

```json
[
  {
    "tipo_producto": "2",
    "numero_sorteo": 789,
    "id_sorteo": "2-789",
    "clave_edicion_producto": 1011,
    "boletos_en_semana_cierre_objetivo": 1,
    "boletos_fuera_semana_cierre_objetivo": 0
  }
]
```

## Métricas Calculadas

### Métricas Históricas
- **hist_boletos_pre_cierre**: Suma de boletos comprados en períodos pre-cierre de ediciones anteriores
- **hist_boletos_semana_cierre**: Suma de boletos comprados en semanas de cierre de ediciones anteriores
- **hist_ediciones_distintas**: Cantidad de ediciones únicas donde el cliente compró

### Métricas de Última Edición
- **ultima_boletos_pre_cierre**: Boletos comprados en pre-cierre de la edición actual
- **ultima_boletos_semana_cierre**: Boletos comprados en semana de cierre de la edición actual
- **compro_ultima_edicion**: Indicador booleano si compró o no en la última edición

### Métricas de Otros Productos
- **otros_boletos_en_semana_cierre_del_producto**: Boletos de otros tipos en la semana de cierre del producto principal
- **otros_boletos_fuera_semana_cierre_del_producto**: Boletos de otros tipos fuera de la semana de cierre
- **otros_tipos_producto_distintos**: Variedad de productos diferentes comprados

## Ventanas Temporales

### Concepto de Ventana

Cada ventana representa una edición de sorteo dentro del último año:

- **Ventana 0**: Edición más reciente (última celebrada)
- **Ventana -1**: Edición anterior
- **Ventana -2, -3, ...**: Ediciones progresivamente más antiguas

### Construcción de Ventanas

```python
windows_map[tipo_producto] = [
    (0, numero_sorteo_n, fecha_inicio_n, fecha_fin_n, clave_n),     # Más reciente
    (-1, numero_sorteo_n-1, fecha_inicio_n-1, fecha_fin_n-1, clave_n-1), # Anterior
    ...
]
```

Las ventanas se construyen:
1. Por tipo de producto
2. Ordenadas por numero_sorteo descendente
3. Limitadas al último año (365 días desde hoy)
4. Solo incluyen ediciones con fecha_fin <= hoy

## Reglas de Negocio

### Clasificación de Canibalización

La clasificación sigue esta lógica jerárquica:

```python
if not compro_ultima and (otros_bol_cierre + otros_bol_fuera) > 0:
    if otros_bol_cierre > 0:
        cani = 'PERDIDO_A_OTRO_EN_CIERRE'
    else:
        cani = 'PERDIDO_A_OTRO_FUERA_CIERRE'
elif not compro_ultima:
    cani = 'CHURN_TOTAL'
elif compro_ultima and (otros_bol_cierre + otros_bol_fuera) > 0:
    cani = 'RETENIDO_Y_OTROS'
else:
    cani = 'RETENIDO'
```

### Filtros de Validez

Los registros deben cumplir:
1. `fecha_rotulacion_finalizacion` no nula
2. `tipo_producto` no nulo
3. `numero_sorteo` no nulo

### Filtro de Clientes Recurrentes

En las versiones "FilterClients":
- Solo se incluyen clientes con `hist_ediciones_distintas >= 2`
- Esto asegura análisis solo de clientes con historial de compras múltiples