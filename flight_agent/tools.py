from datetime import date
from uuid import uuid4

from .data import AIRLINES, CITIES


def result(status, message, *, missing=(), alternatives=()):
    return dict(status=status, message=message, missing_fields=list(missing), alternatives=list(alternatives))


def describe(row):
    return (
        f'{row["number"]} ({AIRLINES[row["airline"]]}) · {CITIES[row["origin"]]} → '
        f'{CITIES[row["destination"]]} · {row["flight_date"]:%d.%m.%Y} '
        f'{row["departure"]:%H:%M} · {row["status"]} · мест: {row["seats"]} · '
        f'без багажа {row["price_light"]:,} ₽ · с багажом {row["price_baggage"]:,} ₽'.replace(",", " ")
    )


def describe_fare(price, baggage):
    suffix = "с багажом" if baggage else "без багажа"
    return f"{price:,}".replace(",", " ") + f" ₽ {suffix}"


def flight_status(store, number, flight_date=None):
    with store.connect() as db:
        rows = db.execute(
            """SELECT * FROM flights WHERE number = %s
               AND (%s::date IS NULL OR flight_date = %s) ORDER BY flight_date""",
            (number, flight_date, flight_date),
        ).fetchall()
    if not rows:
        return result("not_found", "Рейс не найден в деморасписании."), None
    if len(rows) > 1:
        return list_result(rows, "clarify", "Уточните дату вылета.", missing=["flight_date"]), None
    row = rows[0]
    return result("found", describe(row)), {"number": number, "flight_date": row["flight_date"].isoformat()}


def list_result(rows, status, message, *, missing=()):
    answer = result(status, message, missing=missing,
                    alternatives=(f"{i}. {describe(row)}" for i, row in enumerate(rows, 1)))
    answer["offers"] = [
        {"offer_id": (position - 1) * 2 + int(baggage) + 1, "flight_position": position,
         "number": row["number"], "flight_date": row["flight_date"].isoformat(),
         "departure": row["departure"].strftime("%H:%M"),
         "origin": row["origin"], "destination": row["destination"], "status": row["status"], "seats": row["seats"],
         "fare_price": row[column], "fare_baggage": baggage,
         "label": f'{row["number"]} · {row["flight_date"]:%d.%m.%Y} {row["departure"]:%H:%M} · '
                  f'{describe_fare(row[column], baggage)}'}
        for position, row in enumerate(rows, 1)
        for baggage, column in ((False, "price_light"), (True, "price_baggage"))
    ]
    return answer


def list_flights(store, fields):
    with store.connect() as db:
        rows = db.execute(
            """SELECT * FROM flights
               WHERE (%(number)s::text IS NULL OR number = %(number)s)
               AND (%(flight_date)s::date IS NULL OR flight_date = %(flight_date)s)
               AND (%(airline)s::text IS NULL OR airline = %(airline)s)
               AND (%(origin)s::text IS NULL OR origin = %(origin)s)
               AND (%(destination)s::text IS NULL OR destination = %(destination)s)
               ORDER BY flight_date, departure, number""",
            dict.fromkeys(("number", "flight_date", "airline", "origin", "destination")) | fields,
        ).fetchall()
    if not rows:
        return result("not_found", "В деморасписании нет рейсов по этим условиям.")
    return list_result(rows, "listed", f"Рейсы в деморасписании · {len(rows)}")


def my_bookings(store, account_id):
    with store.connect() as db:
        rows = db.execute(
            """SELECT b.id AS booking_id, b.fare_price, b.fare_baggage, b.cancelled_at, f.*
               FROM bookings b JOIN flights f ON f.id = b.flight_id
               WHERE b.account_id = %s ORDER BY f.flight_date, f.departure, b.id""",
            (account_id,),
        ).fetchall()
    if not rows:
        return result("no_bookings", "У вас пока нет демоброней.")
    lines = []
    for row in rows:
        line = f'{row["booking_id"]} · {describe(row)}'
        if row["cancelled_at"] is not None:
            line += " · бронь отменена"
        if row["fare_price"] is not None:
            line += f' · забронирован тариф: {describe_fare(row["fare_price"], row["fare_baggage"])}'
        lines.append(line)
    return result("bookings", f"Ваши демоброни · {len(rows)}", alternatives=lines)


def find_cancellable(store, account_id, *, booking_id=None, number=None, flight_date=None):
    with store.connect() as db:
        return db.execute(
            """SELECT b.id AS booking_id, f.*
               FROM bookings b JOIN flights f ON f.id = b.flight_id
               WHERE b.account_id = %(account_id)s AND b.cancelled_at IS NULL
                 AND (%(booking_id)s::uuid IS NULL OR b.id = %(booking_id)s)
                 AND (%(number)s::text IS NULL OR f.number = %(number)s)
                 AND (%(flight_date)s::date IS NULL OR f.flight_date = %(flight_date)s)
               ORDER BY f.flight_date, f.departure, b.id""",
            dict(account_id=account_id, booking_id=booking_id, number=number, flight_date=flight_date),
        ).fetchall()


def cancel_booking(store, account_id, booking_id):
    with store.connect() as db:
        booking = db.execute(
            "SELECT flight_id FROM bookings WHERE id = %s AND account_id = %s",
            (booking_id, account_id),
        ).fetchone()
        if booking is None:
            return result("no_booking_to_cancel", "Активная бронь не найдена.")
        flight_id = booking["flight_id"]
        db.execute("SELECT id FROM flights WHERE id = %s FOR UPDATE", (flight_id,))
        updated = db.execute(
            """UPDATE bookings SET cancelled_at = now()
               WHERE id = %s AND account_id = %s AND cancelled_at IS NULL RETURNING id""",
            (booking_id, account_id),
        ).fetchone()
        if updated is None:
            return result("no_booking_to_cancel", "Эта бронь уже отменена.")
        db.execute("UPDATE flights SET seats = seats + 1 WHERE id = %s", (flight_id,))
    return result("booking_cancelled", f"Бронь {booking_id} отменена. Место снова доступно.")


def resolve_flight(db, fields):
    requested = date.fromisoformat(fields["flight_date"]) if fields.get("flight_date") else None
    if number := fields.get("number"):
        rows = db.execute("SELECT * FROM flights WHERE number = %s ORDER BY flight_date",
                          (number,)).fetchall()
    else:
        rows = db.execute(
            """SELECT * FROM flights WHERE airline = %s AND origin = %s AND destination = %s
               ORDER BY flight_date, departure, number""",
            (fields["airline"], fields["origin"], fields["destination"]),
        ).fetchall()
    if not rows:
        if number:
            return None, result("not_found", "Рейс не найден в деморасписании.")
        return None, result("no_route", "У выбранной авиакомпании нет рейсов по этому маршруту.")
    candidates = [r for r in rows if requested is None or r["flight_date"] == requested]
    if not candidates:
        closest = sorted(rows, key=lambda r: (
            abs((r["flight_date"] - requested).days), r["flight_date"] < requested,
            r["departure"], r["number"],
        ))
        return None, list_result(closest[:3], "no_date", "На выбранную дату рейсов нет. Вот ближайшие.")
    if len(candidates) > 1:
        missing = "flight_date" if number else "number"
        return None, list_result(candidates, "clarify", "Выберите один из указанных рейсов.", missing=[missing])
    return candidates[0], None


def unavailable(row, quantity=1):
    if row["status"] == "отменён":
        return result("cancelled", f"Рейс отменён. {describe(row)}")
    if not row["seats"]:
        return result("sold_out", f"На рейсе нет свободных мест. {describe(row)}")
    if row["seats"] < quantity:
        return result("insufficient_seats", f'Нужно мест: {quantity}. Доступно: {row["seats"]}. {describe(row)}')
    return None


def changed_fare(row, fields):
    if "fare_price" not in fields:
        return None
    column = "price_baggage" if fields["fare_baggage"] else "price_light"
    if row[column] != fields["fare_price"]:
        return result("fare_changed", f"Тариф изменился. Запросите список рейсов ещё раз. {describe(row)}")
    return None


def check_booking(store, fields):
    with store.connect() as db:
        row, problem = resolve_flight(db, fields)
    return row, problem or unavailable(row, fields.get("quantity", 1)) or changed_fare(row, fields)


def book_ticket(store, account_id, fields, passenger_name, passport_digest):
    return book_passengers(store, account_id, fields, [(passenger_name, passport_digest)])


def book_passengers(store, account_id, fields, passengers):
    quantity = fields.get("quantity", 1)
    if len(passengers) != quantity or quantity < 1:
        raise ValueError("Количество пассажиров должно совпадать с количеством билетов.")
    digests = [digest for _, digest in passengers]
    if any(digest is None or len(digest) != 32 for digest in digests):
        return result("invalid_passport", "Паспорт РФ должен содержать 10 цифр.", missing=["passport"])
    if len(set(digests)) != quantity:
        return result("invalid_passport", "Укажите разные паспорта для пассажиров.", missing=["passport"])
    with store.connect() as db:
        flight, problem = resolve_flight(db, fields)
        if problem:
            return problem
        row = db.execute("SELECT * FROM flights WHERE id = %s FOR UPDATE", (flight["id"],)).fetchone()
        existing = db.execute(
            "SELECT id, account_id FROM bookings WHERE flight_id = %s AND passport_digest = ANY(%s::bytea[]) "
            "AND cancelled_at IS NULL ORDER BY id", (row["id"], digests),
        ).fetchone()
        if existing:
            message = "На этот рейс уже есть бронь для данного пассажира."
            if existing["account_id"] == account_id:
                message += f' Номер брони: {existing["id"]}.'
            return result("duplicate", message)
        if problem := unavailable(row, quantity) or changed_fare(row, fields):
            return problem
        booking_ids = [uuid4() for _ in passengers]
        with db.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO bookings (id, account_id, flight_id, passenger_name, passport_digest, "
                "fare_price, fare_baggage) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(booking, account_id, row["id"], name, digest, fields.get("fare_price"), fields.get("fare_baggage"))
                 for booking, (name, digest) in zip(booking_ids, passengers, strict=True)],
            )
        db.execute("UPDATE flights SET seats = seats - %s WHERE id = %s", (quantity, row["id"]))
    message = (f"Демобронь подтверждена. Номер: {booking_ids[0]}." if quantity == 1
               else f'Демобронь подтверждена. Мест: {quantity}. Номера: {", ".join(map(str, booking_ids))}.')
    if "fare_price" in fields:
        message += f' Тариф: {describe_fare(fields["fare_price"], fields["fare_baggage"])}.'
        if quantity > 1:
            total = f'{fields["fare_price"] * quantity:,}'.replace(",", " ")
            message += f" Итого: {total} ₽."
    return result("booked", f"{message} {describe({**row, 'seats': row['seats'] - quantity})}")
