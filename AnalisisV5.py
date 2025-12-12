import os
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from datetime import datetime, timedelta

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


# =========================
# Configuración de Logging
# =========================
LOG_DIR = Path("/home/sam.salinas/PythonProjects/H3Canibalization/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "h3canibalization.log"

logger = logging.getLogger("h3canibalization")
logger.setLevel(logging.INFO)

# Evitar handlers duplicados si se reimporta
if not logger.handlers:
    # Rotating file: ~10 MB x 5
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)


def _df_stats(df: pd.DataFrame, name: str) -> str:
    try:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        return f"{name}: shape={df.shape}, memory≈{mem_mb:.2f} MB, cols={list(df.columns)}"
    except Exception as e:
        return f"{name}: shape={df.shape}, (mem n/a: {e})"


def _time_block(label: str):
    """Context manager simple para medir tiempos de etapas."""
    class _Timer:
        def __enter__(self):
            self.t0 = perf_counter()
            logger.info(f"▶️  Iniciando: {label}")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = perf_counter() - self.t0
            if exc:
                logger.exception(f"❌ Error en etapa '{label}' (duración {dt:.2f}s): {exc}")
            else:
                logger.info(f"✅ Finalizado: {label} (duración {dt:.2f}s)")
    return _Timer()


def main():
    try:
        logger.info("======== EJECUCIÓN H3 CANIBALIZATION - INICIO ========")

        # -------------------------
        # Cargar datos originales
        # -------------------------
        with _time_block("Carga CSV originales"):

            logger.info("Cargando dfDetalleCompra desde BigQuery...")

            KEY_FILE_LOCATION = "/home/sam.salinas/PythonProjects/H3Canibalization/sorteostec-ml-5f178b142b6f.json"
            credentials = service_account.Credentials.from_service_account_file(KEY_FILE_LOCATION)
            client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            sql = """
            SELECT
            clave_detalle_compra,
            agrupador_interno_compra, 
            clave_persona,
            clave_rotulo,
            clave_edicion_producto,
            fecha_rotulacion_finalizacion,
            clave_boleto_unico,
            clave_boleto_intranet,
            numero_boleto
            FROM
            `sorteostec-ml.ml_siteconversion.detalle_compra`
            """
            dfDetalleCompra = client.query(sql).to_dataframe()
            dfDetalleCompra.to_csv("/home/sam.salinas/PythonProjects/H3Canibalization/detalle_compra.csv",header=True,index=False)

            logger.info("Cargando dfSorteo desde BigQuery...")
            
            sql = """
            SELECT
            *
            FROM
            `sorteostec-ml.ml_siteconversion.sorteo`
            """
            dfSorteo = client.query(sql).to_dataframe()
            dfSorteo.to_csv("/home/sam.salinas/PythonProjects/H3Canibalization/sorteo.csv",header=True,index=False)

            detalle_path = r'/home/sam.salinas/PythonProjects/H3Canibalization/detalle_compra.csv'
            sorteo_path = r"/home/sam.salinas/PythonProjects/H3Canibalization/sorteo.csv"
            dfDetalleCompra = pd.read_csv(detalle_path)
            dfSorteo = pd.read_csv(sorteo_path, dtype=str)
            logger.info(_df_stats(dfDetalleCompra, "dfDetalleCompra"))
            logger.info(_df_stats(dfSorteo, "dfSorteo (raw, dtype=str)"))

        # -------------------------
        # Convertir tipos de datos necesarios
        # -------------------------
        with _time_block("Conversión de tipos dfSorteo"):
            dfSorteo["clave_edicion_producto"] = dfSorteo["clave_edicion_producto"].astype(int)
            dfSorteo["tipo_producto"] = dfSorteo["tipo_producto"].dropna().astype(float).astype(int)
            dfSorteo["numero_sorteo"] = dfSorteo["numero_sorteo"].dropna().astype(float).astype(int)
            logger.info("Tipos tras conversión: %s", dfSorteo.dtypes.to_dict())

        # -------------------------
        # Convertir fecha_fin a datetime
        # -------------------------
        with _time_block("Parseo de fechas clave en dfSorteo"):
            dfSorteo['fecha_fin'] = pd.to_datetime(dfSorteo['fecha_fin'])
            nan_fin = dfSorteo['fecha_fin'].isna().sum()
            logger.info("dfSorteo.fecha_fin NaT=%d", nan_fin)

        # -------------------------
        # Filtrar 2 años
        # -------------------------
        with _time_block("Filtro de sorteos por ventana de 2 años"):
            fecha_actual = datetime.now()
            fecha_dos_anios_atras = fecha_actual - timedelta(days=730)  # 2 años
            logger.info("Ventana: (%s, %s)", fecha_dos_anios_atras, fecha_actual)

            dfSorteoFiltrado = dfSorteo.loc[
                (dfSorteo["fecha_fin"] < fecha_actual) &
                (dfSorteo["fecha_fin"] > fecha_dos_anios_atras)
            ].copy()
            logger.info(_df_stats(dfSorteoFiltrado, "dfSorteoFiltrado"))

        # -------------------------
        # item_completo y merge
        # -------------------------
        with _time_block("Merge detalle_compra x sorteo_filtrado"):
            dfSorteoFiltrado['item_completo'] = (
                dfSorteoFiltrado['desc_sorteo'] + ' ' + dfSorteoFiltrado['numero_sorteo'].astype(str)
            )
            keep_cols = ['clave_edicion_producto', 'item_completo', "tipo_producto", "numero_sorteo"]
            dfDetalleCompraFiltrado = pd.merge(
                dfDetalleCompra,
                dfSorteoFiltrado[keep_cols],
                on='clave_edicion_producto',
                how='inner'
            )
            
            dfDetalleCompraFiltrado.drop(columns=["item_completo", "tipo_producto", "numero_sorteo"], inplace=True)
            logger.info(_df_stats(dfDetalleCompraFiltrado, "dfDetalleCompraFiltrado"))

        # -------------------------
        # Guardar archivos filtrados
        # -------------------------
        with _time_block("Guardar CSVs filtrados 2 años"):
            out_detalle = r'/home/sam.salinas/PythonProjects/H3Canibalization/detalle_compra_2_años.csv'
            out_sorteo = r'/home/sam.salinas/PythonProjects/H3Canibalization/sorteo_2_años.csv'
            dfDetalleCompraFiltrado.to_csv(out_detalle, index=False)
            dfSorteoFiltrado.to_csv(out_sorteo, index=False)
            logger.info("CSV guardados: %s (rows=%d), %s (rows=%d)",
                        out_detalle, len(dfDetalleCompraFiltrado), out_sorteo, len(dfSorteoFiltrado))
            logger.info("✓ Sorteos filtrados: %d", len(dfSorteoFiltrado))
            logger.info("✓ Registros detalle compra filtrados: %d", len(dfDetalleCompraFiltrado))

        # -------------------------
        # Segunda parte: enriquecimiento y rollup
        # -------------------------
        with _time_block("Carga CSVs filtrados"):
            df_detalle = pd.read_csv(r'/home/sam.salinas/PythonProjects/H3Canibalization/detalle_compra_2_años.csv')
            df_sorteo = pd.read_csv(r'/home/sam.salinas/PythonProjects/H3Canibalization/sorteo_2_años.csv')
            logger.info(_df_stats(df_detalle, "df_detalle"))
            logger.info(_df_stats(df_sorteo, "df_sorteo"))

        # Parsear fechas
        with _time_block("Parseo de fechas (detalle y sorteo)"):
            def parse_fechas(x):
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return pd.to_datetime(x, format=fmt)
                    except (ValueError, TypeError):
                        continue
                return pd.NaT

            df_detalle["fecha_rotulacion_finalizacion"] = df_detalle["fecha_rotulacion_finalizacion"].apply(parse_fechas)
            df_sorteo['fecha_inicio'] = pd.to_datetime(df_sorteo['fecha_inicio'])
            df_sorteo['fecha_fin'] = pd.to_datetime(df_sorteo['fecha_fin'])
            df_sorteo['fecha_celebracion'] = pd.to_datetime(df_sorteo['fecha_celebracion'])

            logger.info("NaT detalle.fecha_rotulacion_finalizacion=%d",
                        df_detalle["fecha_rotulacion_finalizacion"].isna().sum())
            for col in ['fecha_inicio', 'fecha_fin', 'fecha_celebracion']:
                logger.info("NaT sorteo.%s=%d", col, df_sorteo[col].isna().sum())

        # Enriquecer
        with _time_block("Enriquecimiento (merge detalle + sorteo)"):
            df_enriq = df_detalle.merge(
                df_sorteo[['clave_edicion_producto', 'tipo_producto', 'numero_sorteo', 'desc_sorteo',
                           'fecha_inicio', 'fecha_fin', 'fecha_celebracion']].drop_duplicates('clave_edicion_producto'),
                on='clave_edicion_producto',
                how='left'
            )
            logger.info(_df_stats(df_enriq, "df_enriq (post-merge)"))

        # Filtrar válidos
        with _time_block("Filtro de registros válidos en df_enriq"):
            before = len(df_enriq)
            df_enriq = df_enriq[
                df_enriq['fecha_rotulacion_finalizacion'].notna() &
                df_enriq['tipo_producto'].notna() &
                df_enriq['numero_sorteo'].notna()
            ].copy()
            logger.info("Filtrados %d → %d (eliminados=%d)", before, len(df_enriq), before - len(df_enriq))

        # Flags por edición (pre_cierre / semana_cierre) — por fila 
        with _time_block("Etiquetado de ventanas por edición (pre_cierre / semana_cierre)"):
            df_enriq['semana_cierre_ini'] = df_enriq['fecha_fin'] - pd.Timedelta(days=7)
            df_enriq['en_semana_cierre'] = (
                (df_enriq['fecha_rotulacion_finalizacion'] >= df_enriq['semana_cierre_ini']) &
                (df_enriq['fecha_rotulacion_finalizacion'] <= df_enriq['fecha_fin'])
            )
            df_enriq['en_pre_cierre'] = (
                (df_enriq['fecha_rotulacion_finalizacion'] >= df_enriq['fecha_inicio']) &
                (df_enriq['fecha_rotulacion_finalizacion'] < df_enriq['semana_cierre_ini'])
            )
            logger.info("Marcados flags por edición: en_semana_cierre=%d, en_pre_cierre=%d",
                        df_enriq['en_semana_cierre'].sum(), df_enriq['en_pre_cierre'].sum())

        # Normalización mínima de tipos para uso en ventanas (necesario para comparar números)
        with _time_block("Normalización de tipos (numero_sorteo / tipo_producto)"):
            df_sorteo['tipo_producto'] = df_sorteo['tipo_producto'].astype(int)
            df_sorteo['numero_sorteo'] = df_sorteo['numero_sorteo'].astype(int)
            df_sorteo['clave_edicion_producto'] = df_sorteo['clave_edicion_producto'].astype(int)
            df_enriq['tipo_producto'] = df_enriq['tipo_producto'].astype(int)
            df_enriq['numero_sorteo'] = df_enriq['numero_sorteo'].astype(int)
            logger.info("Tipos df_sorteo: %s", df_sorteo.dtypes[['tipo_producto','numero_sorteo','clave_edicion_producto']].to_dict())
            logger.info("Tipos df_enriq: %s", df_enriq.dtypes[['tipo_producto','numero_sorteo']].to_dict())

        # -------------------------
        # Construcción de ventanas (último año) por tipo_producto
        # -------------------------
        with _time_block("Construcción de ventanas por tipo_producto (último año)"):
            hoy = pd.Timestamp.now().normalize()
            cutoff = hoy - pd.Timedelta(days=365)

            ed_cols = ['tipo_producto', 'numero_sorteo', 'clave_edicion_producto', 'fecha_inicio', 'fecha_fin']
            ed = df_sorteo[ed_cols].sort_values(['tipo_producto', 'numero_sorteo'], ascending=[True, False])

            windows_map = {}  # tipo_producto -> list[(ventana_numero, ns_win, fi_win, ff_win, cep_win)]
            for tp, g in ed.groupby('tipo_producto', sort=False):
                w = []
                i = 0
                for row in g.itertuples(index=False):
                    # Nos detenemos cuando la edición candidata queda fuera del último año
                    if row.fecha_fin < cutoff:
                        break
                    # Solo consideramos ediciones con fecha_fin <= hoy (ya filtrado arriba, pero lo dejamos claro)
                    if row.fecha_fin <= hoy:
                        ventana_numero = -i  # 0 para la más reciente, -1 para la anterior, etc.
                        w.append((
                            ventana_numero,
                            int(row.numero_sorteo),
                            row.fecha_inicio,
                            row.fecha_fin,
                            int(row.clave_edicion_producto)
                        ))
                        i += 1
                if w:
                    windows_map[int(tp)] = w

            logger.info("Ventanas construidas para %d tipos_producto", len(windows_map))

        # -------------------------
        # Función para contar boletos (idéntica a tu lógica)
        # -------------------------
        def count_tickets(df_group):
            if 'numero_boleto' in df_group.columns and df_group['numero_boleto'].notna().any():
                return df_group['numero_boleto'].nunique()
            return len(df_group)

        # -------------------------
        # Agregación por persona×producto y por cada ventana
        # -------------------------
        with _time_block("Agregación por (clave_persona, tipo_producto) y ventanas"):
            rollup_list = []
            grp_iter = df_enriq.groupby(['clave_persona', 'tipo_producto'])
            total_grupos = grp_iter.ngroups
            logger.info("Total de grupos persona×producto: %d", total_grupos)

            for idx, ((persona, tipo), df_grp) in enumerate(grp_iter, start=1):
                if idx % 1000 == 0:
                    logger.info("Progreso grupos: %d / %d (%.1f%%)", idx, total_grupos, 100 * idx / total_grupos)

                # Si no hay ventanas para este tipo, continuar
                if int(tipo) not in windows_map:
                    continue

                # Cache para “otros” del cliente
                df_cliente = df_enriq[df_enriq['clave_persona'] == persona]

                for (ventana_num, ns_win, fi_win, ff_win, cep_win) in windows_map[int(tipo)]:
                    # “Como si” la tabla llegara hasta la edición ns_win (excluye futuras)
                    df_grp_win = df_grp[df_grp['numero_sorteo'] <= ns_win]

                    # --- HISTÓRICO (ediciones anteriores a la última de la ventana) ---
                    df_hist = df_grp_win[df_grp_win['numero_sorteo'] < ns_win]
                    hist_pre = count_tickets(df_hist[df_hist['en_pre_cierre']]) if not df_hist.empty else 0
                    hist_cierre = count_tickets(df_hist[df_hist['en_semana_cierre']]) if not df_hist.empty else 0
                    hist_ediciones = df_hist['numero_sorteo'].nunique() if not df_hist.empty else 0

                    # --- ÚLTIMA EDICIÓN (la de la ventana) ---
                    df_ult = df_grp_win[df_grp_win['numero_sorteo'] == ns_win]
                    ult_pre = count_tickets(df_ult[df_ult['en_pre_cierre']]) if not df_ult.empty else 0
                    ult_cierre = count_tickets(df_ult[df_ult['en_semana_cierre']]) if not df_ult.empty else 0
                    compro_ultima = (ult_pre + ult_cierre) > 0

                    # Datos de la “última” (de la ventana)
                    ultima_ini = fi_win
                    ultima_fin = ff_win
                    numero_sorteo_ultima = ns_win
                    clave_edicion_ultima = cep_win

                    # --- OTROS SORTEOS (tipo distinto) en ventana de la última (de la ventana) ---
                    otros_bol_cierre = 0
                    otros_bol_fuera = 0
                    otros_detalle = []

                    en_ventana = (
                        (df_cliente['fecha_rotulacion_finalizacion'] >= ultima_ini) &
                        (df_cliente['fecha_rotulacion_finalizacion'] <= ultima_fin)
                    )
                    df_otros = df_cliente[en_ventana & (df_cliente['tipo_producto'] != tipo)]

                    if not df_otros.empty:
                        cierre_ini_obj = ultima_fin - pd.Timedelta(days=7)
                        df_otros = df_otros.copy()
                        df_otros['en_cierre_objetivo'] = (
                            (df_otros['fecha_rotulacion_finalizacion'] >= cierre_ini_obj) &
                            (df_otros['fecha_rotulacion_finalizacion'] <= ultima_fin)
                        )

                        for (tp2, ns2), df_g2 in df_otros.groupby(['tipo_producto', 'numero_sorteo']):
                            bol_cierre = count_tickets(df_g2[df_g2['en_cierre_objetivo']])
                            bol_fuera = count_tickets(df_g2[~df_g2['en_cierre_objetivo']])

                            otros_bol_cierre += bol_cierre
                            otros_bol_fuera += bol_fuera

                            otros_detalle.append({
                                'tipo_producto': str(int(tp2)) if pd.notna(tp2) else None,
                                'numero_sorteo': int(ns2) if pd.notna(ns2) else None,
                                'id_sorteo': f"{int(tp2)}-{int(ns2)}" if pd.notna(ns2) else f"{int(tp2)}-None",
                                'clave_edicion_producto': int(df_g2['clave_edicion_producto'].iloc[0]) if pd.notna(df_g2['clave_edicion_producto'].iloc[0]) else None,
                                'boletos_en_semana_cierre_objetivo': int(bol_cierre),
                                'boletos_fuera_semana_cierre_objetivo': int(bol_fuera),
                            })

                    otros_tipos_distintos = len(set(r['tipo_producto'] for r in otros_detalle))

                    # --- DETALLE HISTÓRICO por edición ---
                    hist_detalle = []
                    if not df_hist.empty:
                        for ns_h, df_h in df_hist.groupby('numero_sorteo'):
                            hist_detalle.append({
                                'numero_sorteo': int(ns_h) if pd.notna(ns_h) else None,
                                'tipo_producto': str(int(tipo)),
                                'id_sorteo': f"{int(tipo)}-{int(ns_h)}" if pd.notna(ns_h) else f"{int(tipo)}-None",
                                'clave_edicion_producto': int(df_h['clave_edicion_producto'].iloc[0]) if pd.notna(df_h['clave_edicion_producto'].iloc[0]) else None,
                                'boletos_pre_cierre': int(count_tickets(df_h[df_h['en_pre_cierre']])),
                                'boletos_semana_cierre': int(count_tickets(df_h[df_h['en_semana_cierre']])),
                                'fecha_fin_edicion': df_h['fecha_fin'].iloc[0].strftime('%Y-%m-%d') if pd.notna(df_h['fecha_fin'].iloc[0]) else None
                            })

                    # --- CLASIFICACIÓN DE CANIBALIZACIÓN (misma lógica) ---
                    if not compro_ultima and (otros_bol_cierre + otros_bol_fuera) > 0:
                        cani = 'PERDIDO_A_OTRO_EN_CIERRE' if otros_bol_cierre > 0 else 'PERDIDO_A_OTRO_FUERA_CIERRE'
                    elif not compro_ultima:
                        cani = 'CHURN_TOTAL'
                    elif compro_ultima and (otros_bol_cierre + otros_bol_fuera) > 0:
                        cani = 'RETENIDO_Y_OTROS'
                    else:
                        cani = 'RETENIDO'

                    # --- GUARDAR RESULTADO (agrego ventana_numero) ---
                    rollup_list.append({
                        'ventana_numero': int(ventana_num),  # 0, -1, -2, ...
                        'clave_persona': int(persona) if pd.notna(persona) else None,
                        'tipo_producto': int(tipo) if pd.notna(tipo) else None,
                        'numero_sorteo_ultima': int(numero_sorteo_ultima) if pd.notna(numero_sorteo_ultima) else None,
                        'clave_edicion_producto_ultima': int(clave_edicion_ultima) if pd.notna(clave_edicion_ultima) else None,
                        'ultima_fecha_inicio': ultima_ini.strftime('%Y-%m-%d') if pd.notna(ultima_ini) else None,
                        'ultima_fecha_fin': ultima_fin.strftime('%Y-%m-%d') if pd.notna(ultima_fin) else None,
                        'ultima_semana_cierre_ini': (ultima_fin - pd.Timedelta(days=7)).strftime('%Y-%m-%d') if pd.notna(ultima_fin) else None,
                        'ultima_semana_cierre_fin': ultima_fin.strftime('%Y-%m-%d') if pd.notna(ultima_fin) else None,
                        'hist_boletos_pre_cierre': int(hist_pre),
                        'hist_boletos_semana_cierre': int(hist_cierre),
                        'hist_ediciones_distintas': int(hist_ediciones),
                        'ultima_boletos_pre_cierre': int(ult_pre),
                        'ultima_boletos_semana_cierre': int(ult_cierre),
                        'compro_ultima_edicion': bool(compro_ultima),
                        'otros_boletos_en_semana_cierre_del_producto': int(otros_bol_cierre),
                        'otros_boletos_fuera_semana_cierre_del_producto': int(otros_bol_fuera),
                        'otros_tipos_producto_distintos': int(otros_tipos_distintos),
                        'canibalizacion_clase': cani,
                        'hist_detalle_por_edicion_json': json.dumps(hist_detalle, ensure_ascii=False),
                        'otros_detalle_en_ventana_ultima_json': json.dumps(otros_detalle, ensure_ascii=False),
                    })

            logger.info("Agregación completada. Registros rollup_list: %d", len(rollup_list))

        # Crear DataFrame final + Guardar CSV
        with _time_block("Creación y guardado de rollup_canibalizacion.csv"):
            rollup_df = pd.DataFrame(rollup_list)
            logger.info(_df_stats(rollup_df, "rollup_df"))
            out_rollup = r'/home/sam.salinas/PythonProjects/H3Canibalization/rollup_canibalizacion.csv'
            rollup_df.to_csv(out_rollup, index=False)
            logger.info("Archivo generado: %s (registros=%d)", out_rollup, len(rollup_df))

        # Preparar columnas fecha para BigQuery
        with _time_block("Conversión de columnas de fecha (para BigQuery)"):
            columnas_fecha = [
                'ultima_fecha_inicio',
                'ultima_fecha_fin',
                'ultima_semana_cierre_ini',
                'ultima_semana_cierre_fin'
            ]
            for col in columnas_fecha:
                rollup_df[col] = pd.to_datetime(rollup_df[col], errors='coerce')
                logger.info("NaT en %s: %d", col, rollup_df[col].isna().sum())

        # Configuración BigQuery
        with _time_block("Configuración cliente BigQuery"):
            credentials_path = r'/home/sam.salinas/PythonProjects/H3Canibalization/sorteostec-ml-5f178b142b6f.json'  # PLACEHOLDER
            bq_project = 'sorteostec-ml'  # PLACEHOLDER
            dataset_id = 'h3'  # PLACEHOLDER
            table_id = 'rollup_canibalizacion'  # PLACEHOLDER

            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = bigquery.Client(credentials=credentials, project=bq_project)
            full_table_id = f"{client.project}.{dataset_id}.{table_id}"
            logger.info("BigQuery destino: %s | Credenciales: %s", full_table_id, credentials_path)

        # Cargar a BigQuery
        with _time_block("Carga a BigQuery (WRITE_TRUNCATE)"):
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            logger.info("Cargando %d registros a %s ...", len(rollup_df), full_table_id)
            job = client.load_table_from_dataframe(rollup_df, full_table_id, job_config=job_config)
            job.result()  # Espera a terminar
            logger.info("Datos cargados exitosamente a BigQuery: %s | job_id=%s | output_rows=%s",
                        full_table_id, job.job_id, getattr(job, "output_rows", "desconocido"))

        logger.info("======== EJECUCIÓN H3 CANIBALIZATION - FIN EXITOSO ========")

    except Exception as e:
        logger.exception("Ejecución abortada por error: %s", e)
        raise


if __name__ == "__main__":
    main()
