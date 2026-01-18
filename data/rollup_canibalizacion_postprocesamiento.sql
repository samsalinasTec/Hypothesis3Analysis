WITH
  t AS (
  SELECT
    t.clave_persona,
    t.clave_edicion_producto_ultima,
    CAST(JSON_VALUE(detalle, '$.clave_edicion_producto') AS INT64) AS clave_edicion_producto_extraida
  FROM
    `sorteostec-ml.h3.rollup_canibalizacion_backup24102025` t,
    UNNEST(JSON_QUERY_ARRAY(t.otros_detalle_en_ventana_ultima_json)) AS detalle
  WHERE
    t.canibalizacion_clase = "PERDIDO_A_OTRO_EN_CIERRE"
    AND CAST(JSON_VALUE(detalle, '$.boletos_en_semana_cierre_objetivo') AS INT64) > 0 )
SELECT
  t.clave_edicion_producto_ultima AS producto_origen_id,
  CONCAT(s_origen.desc_corta_sorteo, " ", CAST(s_origen.numero_sorteo AS INT64)) AS producto_origen_tag,
  t.clave_edicion_producto_extraida AS producto_destino_id,
  CONCAT(s_destino.desc_corta_sorteo, " ", CAST(s_destino.numero_sorteo AS INT64)) AS producto_destino_tag,
  COUNT(DISTINCT t.clave_persona) AS clientes_perdidos
FROM
  t
LEFT JOIN
  `sorteostec-ml.ml_siteconversion.sorteo` s_destino
ON
  t.clave_edicion_producto_extraida = s_destino.clave_edicion_producto
LEFT JOIN
  `sorteostec-ml.ml_siteconversion.sorteo` s_origen
ON
  t.clave_edicion_producto_ultima = s_origen.clave_edicion_producto
GROUP BY
  1,
  2,
  3,
  4
ORDER BY
  producto_origen_tag,
  clientes_perdidos DESC;