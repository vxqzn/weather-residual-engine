from twre.db.session import get_connection

with get_connection() as conn, conn.cursor() as cur:
    for t, col in [("weather_observations", "date"), ("forecast_predictions", "target_date"), ("realized_errors", "target_date")]:
        cur.execute(f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {t};")
        c, mn, mx = cur.fetchone()
        print(f"{t:22} | count: {c} | range: {mn} -> {mx}")