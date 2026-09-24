from twre.db.session import get_connection

with get_connection() as conn, conn.cursor() as cur:
    for t in ["weather_observations", "forecast_predictions", "realized_errors"]:
        cur.execute(f"SELECT COUNT(*), MIN(COALESCE(date, target_date)), MAX(COALESCE(date, target_date)) FROM {t};")
        c, mn, mx = cur.fetchone()
        print(f"{t:22} | count: {c} | range: {mn} -> {mx}")